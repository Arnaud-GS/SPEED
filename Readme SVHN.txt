# The code for learning (except improved GAN) and for DP analysis was forked from https://github.com/tensorflow/privacy/tree/master/research/pate_2017 (described in the following paper : https://arxiv.org/abs/1610.05755)
# and adapted to our work, especially for BHBC framework (flag distributed_noise = True)

# The code for improved GAN was forked and adapted from https://github.com/openai/improved-gan

Here is the workflow of the whole learning process with HE layer:
- Train teachers
## this generates the 250 teacher models
## max_steps = 2000
## deeper
## batch-size = 64
## learning_rate=0.08
seq 0 249 | parallel -j8 CUDA_VISIBLE_DEVICES='$(({%}-1))' python train_teachers.py --nb_teachers=250 \
	--teacher_id={} --deeper=True --dataset=svhn --data_dir=data_path \
	--train_dir=path_to_save_teachers --max_steps=2000

- Get the one-hot encodings of the teachers' votes
## deeper=True
python train_student.py --nb_teachers=250 -- only_get_ohe=True --stdnt_share=500 --deeper=True\
	--distributed_noise=<True/False> --nb_successful_teachers=<250/225/175> \
	--dataset=svhn --data_dir=data_path --teachers_dir=path_where_teachers_are_saved \
	--train_dir=path_to_save_student --ohe_file=file_to_save_one_hot_encodings \
	--teachers_max_steps=2000
# for HBC framework, the flag --lap_noises_file=file_to_save_laplace_noises
# can be added to save an array of <number of teachers> * <number of classes> noise values
# that will be used by the HE layer

# Note that the previous line will return an assertion error (because of the only_get_ohe flag)
# since the train_student function was stopped before ending the training.
# Indeed, we only need the execution to generate the .txt files.

- Perform homomorphic aggregation (see makefile) with one-hot encodings as input
# For HBC framework, add Laplace noise to the one-hot encodings via ciphertext vs cleartext addition

- Convert .txt files output by homomorphic aggregation into .npy files
python file_conversion.py --nb_teachers=250 --dataset=svhn --stdnt_share=500 \
	--distributed_noise=<True/False> --nb_successful_teachers=<250/225/175> \
	--ohe_after_HE_argmax_dir=path_where_ohe_are_saved --data_after_argmax_dir=path_to_save_labels

- Train the student using improved GAN with npy labels file as input
# lr = 0.0003
# epochs = 600
train_svhn_fm_custom_labels.py --labels=labelfile --epochs=600


Here is the command to execute for the DP analysis:
python analysis.py --counts_file=svhn_250_teachers_labels.npy \
	--distributed_noise=<True/False> --noise_eps=3.3 --ratio_successful_teachers=<1/0.90.7> --max_examples=500

