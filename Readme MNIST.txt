# The code for learning (except improved GAN) and DP analysis was forked from https://github.com/tensorflow/privacy/tree/master/research/pate_2017 (described in the following paper : https://arxiv.org/abs/1610.05755)
# and adapted to our work, especially for BHBC framework (flag distributed_noise = True)

# The code for improved GAN was forked and adapted from https://github.com/openai/improved-gan

Here is the workflow of the whole learning process with HE layer:
- Train teachers
## this generates the 250 teacher models
seq 0 249 | parallel -j8 CUDA_VISIBLE_DEVICES='$(({%} - 1))' python train_teachers.py --nb_teachers=250 --teacher_id={} \
	--dataset=mnist --data_dir=data_path --train_dir=path_to_save_teachers --max_steps=5000

- Get the one-hot encodings of the teachers' votes
python train_student.py --nb_teachers=250 -- only_get_ohe=True --stdnt_share=100 \
	--distributed_noise=<True/False> --nb_successful_teachers=<250/225/175> \
	--dataset=mnist --data_dir=data_path --teachers_dir=path_where_teachers_are_saved \
	--train_dir=path_to_save_student --ohe_file=file_to_save_one_hot_encodings \
	--teachers_max_steps=5000
# for HBC framework, the flag --lap_noises_file=file_to_save_laplace_noises
# can be added to generate an array of <number of teachers> * <number of classes> noise values
# that will be used by the HE layer

# Note that the previous line will return an assertion error (because of the only_get_ohe flag)
# since the train_student function was stopped before ending the training.
# Indeed, we only need the execution to generate the .txt files.

- Perform homomorphic aggregation (see makefile) with one-hot encodings as input
# For HBC framework, add Laplace noise to the one-hot encodings via ciphertext vs cleartext addition

- Convert .txt files output by homomorphic aggregation into .npy files
python file_conversion.py --nb_teachers=250 --dataset=mnist --stdnt_share=100 \
	--distributed_noise=<True/False> --nb_successful_teachers=<250/225/175> \
	--ohe_after_HE_argmax_dir=path_where_ohe_are_saved --data_after_argmax_dir=path_to_save_labels

- Train the student using improved GAN with npy labels file as input
# lr = 0.001
# epochs = 500
train_mnist_fm_custom_labels.py --labels=labelfile --seed_data=1 --count=10 --epochs=500


Here is the command to execute for the DP analysis:
python analysis.py --counts_file=mnist_250_teachers_labels.npy --indices_file=mnist_250_teachers_100_indices_used_by_student.npy \
	--distributed_noise=<True/False> --noise_eps=3.3 --ratio_successful_teachers=<1/0.9/0.7> --max_examples=100

