#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>

#include <time.h>

// I/O
#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>

// TFHE
// - Main TFHE
#include "tfhe_gate.h"
#include "tfhe_functions.h"
// - I/O
#include "tfhe_io.h"
#include "tfhe_generic_streams.h"
// Key switch
#include "tlwe-functions-extra.h"

using namespace std;

/**
* In here, we are going to test argmax operations with given parameters
*/
int main()
{
  // The random seed
  srand(time(NULL));

  /**
  * Problem parameters
  */

  /*
  * The size of the label vectors
  */
  const int d = 10;    // "classes"

  /**
  * The number of small model
  */
  const int gamma = 250;   // "teachers"

  /**
  * Printing width
  */
  const int barWidth = 70;

  /**
  * Decide whether to print the time of operation or not
  */
  const int print_time = 0;

  /**
  * Differential privacy amplitude
  */
  const int amplitude = 900;

  /**
  * The precision used for input values
  */
  const int dataPrecision = 1000;

  /**
  * Number of data vectors with which to test
  */
  const int number_of_testing_loops = 10;

  /**
  * Open file for reading the noise when in the non-distributed case
  */

  FILE * g;
  g = fopen("<INSERT NOISE FILE PATH HERE>", "r");

  /**
  * Open file for reading the data from the teachers
  */

  FILE * f;
  f = fopen("<INSERT VECTOR FILE PATH HERE>", "r");

  /**
  * The file to write into
  */

  FILE * h;
  h = fopen("<INSERT OUTPUT FILE PATH HERE>", "w+");

  /**
  * Initialize TFHE parameter structures
  */

  // Bootstrapping parameters
  const int boot_l = 6;
  const int boot_Bgbits = 6;

  // Initial parameters
  const double init_N = 1024;
  const double init_alpha = 1e-9;

  const TLweParams<Torus64> * init_tlwe_params = new_TLweParams<Torus64>(init_N, 1, init_alpha, init_alpha);
  const TGswParams<Torus64> * init_tgsw_params = new_TGswParams<Torus64>(boot_l, boot_Bgbits, init_tlwe_params);
  const LweParams<Torus64> * init_lwe_params = &init_tlwe_params->extracted_lweparams;

  // Output parameters
  const double out_N = init_N;
  const double out_alpha = init_alpha;

  const TLweParams<Torus64> * out_tlwe_params = new_TLweParams<Torus64>(out_N, 1, out_alpha, out_alpha);
  const TGswParams<Torus64> * out_tgsw_params = new_TGswParams<Torus64>(boot_l, boot_Bgbits, out_tlwe_params);
  const LweParams<Torus64> * out_lwe_params = &out_tlwe_params->extracted_lweparams;

  /**
  * Print the parameters
  */

  printf("TFHE parameters:\n");
  printf("-- N: %d \n", (int) init_N);
  printf("-- alpha: %.2e\n", init_alpha);
  std::cout << '\n';

  /**
  * Initialize TFHE key structures
  */

  // Initial key
  // The one with which everything is encrypted at first
  TGswKey<Torus64> * init_tgsw_key = new_TGswKey<Torus64>(init_tgsw_params);
  tGswKeyGen(init_tgsw_key);

  TLweKey<Torus64> * init_tlwe_key = &(init_tgsw_key->tlwe_key);
  LweKey<Torus64> * init_lwe_key = new_LweKey<Torus64>(init_lwe_params);
  tLweExtractKey(init_lwe_key, init_tlwe_key);

  // The second key, which is in this case an intermediate key
  TGswKey<Torus64> * out_tgsw_key = new_TGswKey<Torus64>(out_tgsw_params);
  tGswKeyGen(out_tgsw_key);

  TLweKey<Torus64> * out_tlwe_key = &(out_tgsw_key->tlwe_key);
  LweKey<Torus64> * out_lwe_key = new_LweKey<Torus64>(out_lwe_params);
  tLweExtractKey(out_lwe_key, out_tlwe_key);

  // Bootstrapping key (the t and base don't matter if we don't use a key-switch key)
  // const int switch_t = 14;
  const int switch_t = 4;
  const int switch_base = 5;

  // Bootstrapping key #1
  LweBootstrappingKey<Torus64> * boot_key = new_LweBootstrappingKey<Torus64>(switch_t , switch_base, init_lwe_params, out_tgsw_params);
  tfhe_createLweBootstrappingKey(boot_key, init_lwe_key, out_tgsw_key);
  LweBootstrappingKeyFFT<Torus64> * boot_key_fft = new_LweBootstrappingKeyFFT<Torus64>(boot_key);

  // Bootstrapping key #2
  LweBootstrappingKey<Torus64> * boot_key2 = new_LweBootstrappingKey<Torus64>(switch_t , switch_base, out_lwe_params, init_tgsw_params);
  tfhe_createLweBootstrappingKey(boot_key2, out_lwe_key, init_tgsw_key);
  LweBootstrappingKeyFFT<Torus64> * boot_key_fft2 = new_LweBootstrappingKeyFFT<Torus64>(boot_key2);

  /**
  * Global useful things
  */

  // Temps
  Torus64 torusTemp;
  int64_t intTemp;
  double doubleTemp;
  float floatTemp;
  LweSample<Torus64> * lweInTemp = new_LweSample(init_lwe_params);
  LweSample<Torus64> * lweOutTemp = new_LweSample(out_lwe_params);

  /**
  * Initialize all of the shell structures
  */

  // Time
  clock_t start;
  double time_comp = 0;
  double time_encryption = 0;
  double time_sum = 0;
  double time_delta = 0;
  double time_delta_sum = 0;
  double time_decryption = 0;
  double time_final_boot = 0;
  double time_key_switch = 0;

  // For progression percentage
  int pos;
  float progress;
  int total_loops;
  int nb_loops;

  // The initial, clear label data
  // There are "gamma" lines, one for every small model
  // There are "d" columns, one for every label
  int ** labels = (int **) malloc(sizeof(int*)*gamma);
  for (int i = 0; i < gamma; i++)
    labels[i] = (int *) malloc(sizeof(int)*d);

  // The laplacian noise
  int * laplacian_noise = (int *) malloc(sizeof(int)*d);

  // The summed labels
  // Used to compute true_argmax and check the results at the end
  int * summed_labels = (int *) malloc(sizeof(int)*d);

  // Used to store the maximum value
  int max_value;

  // An index used for random attribution of 1 value
  int index;

  // Moduluses used for torus encoding
  int init_modulus;
  int delta_modulus;
  int final_modulus;

  // The TFHE equivalents for te initial labels
  LweSample<Torus64> *** lweLabels = (LweSample<Torus64> ***) malloc(sizeof(LweSample<Torus64> **)*gamma);
  for (int i = 0; i < gamma; i++)
  {
    lweLabels[i] = (LweSample<Torus64> **) malloc(sizeof(LweSample<Torus64> *)*d);
    for (int j = 0; j < d; j++)
      lweLabels[i][j] = new_LweSample(init_lwe_params);
  }

  // The TFHE summed labels
  LweSample<Torus64> ** lweSummedLabels = (LweSample<Torus64> **) malloc(sizeof(LweSample<Torus64> *)*d);
  for (int i = 0; i < d; i++)
    lweSummedLabels[i] = new_LweSample(init_lwe_params);

  // The amplitude ciphertext
  LweSample<Torus64> * lweAmplitude = new_LweSample(init_lwe_params);

  // The difference between the labels
  LweSample<Torus64> * lweDifference = new_LweSample(init_lwe_params);

  // The TFHE delta matrix (d*d)
  // Here we have a full matrix because we are lazy. We could just allocate half of the matrix.
  // We're still not going to compute all of them because we are not that lazy..
  LweSample<Torus64> *** lweDeltas = (LweSample<Torus64> ***) malloc(sizeof(LweSample<Torus64> **)*d);
  for (int i = 0; i < d; i++)
  {
    lweDeltas[i] = (LweSample<Torus64> **) malloc(sizeof(LweSample<Torus64> *)*d);
    for (int j = 0; j < d; j++)
      lweDeltas[i][j] = new_LweSample(out_lwe_params);
  }

  // The shells for the positive and negative default torus values
  Torus64 positive;
  Torus64 negative;

  // The noiseless ciphertext used to operate rotations
  LweSample<Torus64> * lweRotation = new_LweSample(out_lwe_params);

  // The TFHE summed deltas
  LweSample<Torus64> ** lweSummedDeltas = (LweSample<Torus64> **) malloc(sizeof(LweSample<Torus64> *)*d);
  for (int i = 0; i < d; i++)
    lweSummedDeltas[i] = new_LweSample(out_lwe_params);

  // The TFHE key-switched argmax results
  LweSample<Torus64> ** lweSwitchedDeltaSum = (LweSample<Torus64> **) malloc(sizeof(LweSample<Torus64> *)*d);
  for (int i = 0; i < d; i++)
    lweSwitchedDeltaSum[i] = new_LweSample(init_lwe_params);

  // The TFHE argmax results
  LweSample<Torus64> ** lweArgmax = (LweSample<Torus64> **) malloc(sizeof(LweSample<Torus64> *)*d);
  for (int i = 0; i < d; i++)
    lweArgmax[i] = new_LweSample(out_lwe_params);

  // The clear results
  int * argmax = (int *) malloc(sizeof(int)*d);

  // The true clear results
  int * true_argmax = (int *) malloc(sizeof(int)*d);

  // Testing array
  int * testingArray = (int *) malloc(sizeof(int)*d);

  // Values to assess the classification rate
  int classification_hits = 0;    // set at 0 increments for every good classification
  int summedResults;
  int noresult_occurences = 0;    // increments when there is no classification

  /**
  * Print the dimensions
  */

  std::cout << "The testing data size:" << '\n';
  std::cout << "-- " << gamma << " teachers" << '\n';
  std::cout << "-- " << d << " classes" << '\n';
  std::cout << "-- " << number_of_testing_loops << " examples" << '\n';
  std::cout << '\n';


  /**
  * Loop parameters
  */

  // For progression percentage
  total_loops = number_of_testing_loops;
  nb_loops = 0;

  for (int count = 0; count < number_of_testing_loops; count++)
  { // Start of the big loop

    /**
    * Read the labels
    */

    // The "labels" matrix
    for (int i = 0; i < gamma; i++)
    {
      for (int j = 0; j < d; j++)
      {
        fscanf(f, "%f ", &floatTemp);
        labels[i][j] = (int) floatTemp*dataPrecision;
      }
    }

    /**
    * Read the noise in the non-distributed case
    */

    if (distributed == 0)
    {
      for (int j = 0; j < d; j++)
      {
        fscanf(g, "%f ", &floatTemp);
        laplacian_noise[j] = (int) floatTemp*dataPrecision;
      }
    }

    /*
    * Compute the true argmax value in true argmax
    * if there are several max values, all of the indexes associated with them will have a 1 and the others a 0
    */

    // Compute the sum of the labels
    for (int j = 0; j < d; j++)
    {
      // In the case where the noise is not distributed we have to add it here for clear argmax computation
      if (distributed == 0)
        summed_labels[j] = laplacian_noise[j];
      else
        summed_labels[j] = 0;   // Set initial value at 0
      for (int i = 0; i < gamma; i++)
        summed_labels[j] += labels[i][j];
    }

    // First loop to get the max value
    max_value = 0;
    for (int j = 0; j < d; j++)
    {
      if (summed_labels[j] > max_value)
        max_value = summed_labels[j];
    }

    // Loop again to set at 1 for max values
    for (int j = 0; j < d; j++)
    {
      if (summed_labels[j] == max_value)
        true_argmax[j] = 1;
      else
        true_argmax[j] = 0;
    }

    /**
    * TEST
    * print our the labels for every model
    */

    // printf("Labels:\n");
    // // for (int j = 0; j < gamma; j++)
    // for (int j = 0; j < 1; j++)
    // {
    //   printf("Model %d: [", j);
    //   for (int i = 0; i < d-1; i++)
    //     printf("%d, ", labels[j][i]);
    //   printf("%d]\n", labels[j][d-1]);
    // }

    /**
    * Change everything into ciphertexts
    */

    if (print_time == 1)
      std::cout << "- data encryption" << '\n';

    // This is the required modulus for the highest possibe value to be still lower than 1/2
    init_modulus = (2*(gamma+2*amplitude) + 2)*dataPrecision;

    start = clock();

    for (int i = 0; i < gamma; i++)
    {
      for (int j = 0; j < d; j++)
      {
        Torus64 torusTemp = TorusUtils<Torus64>::modSwitchToTorus(labels[i][j], init_modulus);
        LweFunctions<Torus64>::SymEncrypt(lweLabels[i][j], torusTemp, init_alpha, init_lwe_key);
      }
    }

    time_encryption += ( clock() - start ) / (double) CLOCKS_PER_SEC;

    // Print the time that encryption takes for each client
    if (print_time == 1)
      printf("--- %.2e sec.\n", time_encryption/gamma);

    /**
    * Sum it all up in the encrypted domain
    */

    if (print_time == 1)
      std::cout << "- sum of the labels" << '\n';

    start = clock();

    for (int i = 0; i < d; i++)
    {
      LweFunctions<Torus64>::Copy(lweSummedLabels[i], lweLabels[0][i], init_lwe_params);
      for (int j = 1; j < gamma; j++)
        LweFunctions<Torus64>::AddTo(lweSummedLabels[i], lweLabels[j][i], init_lwe_params);
    }

    time_sum += ( clock() - start ) / (double) CLOCKS_PER_SEC;
    time_comp += ( clock() - start ) / (double) CLOCKS_PER_SEC;

    if (print_time == 1)
      printf("--- %.2e sec.\n", time_sum);

    /**
    * TEST
    * print out the summed labels
    */

    // for (int i = 0; i < d; i++)
    // {
    //   torusTemp = LweFunctions<Torus64>::SymDecrypt(lweSummedLabels[i], init_lwe_key, init_modulus);
    //   testingArray[i] = TorusUtils<Torus64>::modSwitchFromTorus(torusTemp, init_modulus);
    // }
    //
    // printf("Summed labels:\n[");
    // for (int i = 0; i < d-1; i++)
    //   printf("%d, ", testingArray[i]);
    // printf("%d]\n", testingArray[d-1]);

    /**
    * Add the amplitude in whatever case
    */

    torusTemp = TorusUtils<Torus64>::modSwitchToTorus(amplitude, init_modulus);
    LweFunctions<Torus64>::NoiselessTrivial(lweAmplitude, torusTemp, init_lwe_params);
    for (int i = 0; i < d; i++)
      LweFunctions<Torus64>::AddTo(lweSummedLabels[i], lweAmplitude, init_lwe_params);

    /**
    * Add the noise in the non-distributed case
    */

    if (distributed == 0)
    {
      for (int i = 0; i < d; i++)
      {
        torusTemp = TorusUtils<Torus64>::modSwitchToTorus(laplacian_noise[i], init_modulus);
        LweFunctions<Torus64>::NoiselessTrivial(lweInTemp, torusTemp, init_lwe_params);
        LweFunctions<Torus64>::AddTo(lweSummedLabels[i], lweInTemp, init_lwe_params);
      }
    }

    /**
    * Compute the deltas
    */

    if (print_time == 1)
      std::cout << "- delta computation" << '\n';

    // Here the modulus is set in the case where we can sum everything once: d < m
    delta_modulus = 4*d-4;
    positive = TorusUtils<Torus64>::modSwitchToTorus(1, delta_modulus);
    negative = TorusUtils<Torus64>::modSwitchToTorus(-1, delta_modulus);

    // Set this one to 1/modulus for rotation pruposes
    LweFunctions<Torus64>::NoiselessTrivial(lweRotation, positive, out_lwe_params);

    start = clock();
    for (int i = 0; i < d; i++)
    {
      for (int j = 0; j < i; j++)
      {
        // Compute label_i - label_j
        LweFunctions<Torus64>::Copy(lweDifference, lweSummedLabels[i], init_lwe_params);
        LweFunctions<Torus64>::SubTo(lweDifference, lweSummedLabels[j], init_lwe_params);

        // If label_i < label_j, then label_i - label_j < 0 and delta_{i,j} = 1/modulus
        // Otherwise delta_{i,j} = -1/modulus
        TfheFunctions<Torus64>::bootstrap_woKS_FFT(lweDeltas[i][j], boot_key_fft, negative, lweDifference);

      }
    }
    // std::cout << std::endl;   // For progression percentage

    // Fill the other half of the matrix with the opposite
    for (int i = 0; i < d; i++)
    {
      for (int j = i+1; j < d; j++)
        LweFunctions<Torus64>::Negate(lweDeltas[i][j], lweDeltas[j][i], out_lwe_params);
    }

    // Add 1/modulus (now values are 2/modulus and 0)
    for (int i = 0; i < d; i++)
    {
      for (int j = 0; j < d; j++)
      {
        if (i != j)
          LweFunctions<Torus64>::AddTo(lweDeltas[i][j], lweRotation, out_lwe_params);
      }
    }
    time_delta += ( clock() - start ) / (double) CLOCKS_PER_SEC;
    time_comp += ( clock() - start ) / (double) CLOCKS_PER_SEC;

    if (print_time == 1)
      printf("--- %.2e sec.\n", time_delta);

    /**
    * TEST
    * print our delta matrix
    */

    // printf("Decrypted deltas:\n");
    // for (int i = 0; i < d; i++)
    // {
    //   printf("[ ");
    //   for (int j = 0; j < d; j++)
    //   {
    //     if (i != j)
    //     {
    //       torusTemp = LweFunctions<Torus64>::SymDecrypt(lweDeltas[i][j], out_lwe_key, delta_modulus);
    //       testingArray[j] = TorusUtils<Torus64>::modSwitchFromTorus(torusTemp, delta_modulus);
    //     }
    //     if (i == j)
    //       testingArray[j] = 0;
    //
    //     printf("%d ", testingArray[j]);
    //   }
    //   printf("]\n");
    // }

    /**
    * Summing the deltas together into lweSummedDeltas
    */

    if (print_time == 1)
      std::cout << "- delta sum" << '\n';

    start = clock();

    // First, initialize with 1/modulus
    for (int j = 0; j < d; j++)
      LweFunctions<Torus64>::Copy(lweSummedDeltas[j], lweRotation, out_lwe_params);

    // Start summing
    for (int i = 0; i < d; i++)
    {
      for (int j = 0; j < d; j++)
      {
        if (i != j)
          LweFunctions<Torus64>::AddTo(lweSummedDeltas[j], lweDeltas[i][j], out_lwe_params);

        // if i = j then do nothing
      }
    }

    time_delta_sum += ( clock() - start ) / (double) CLOCKS_PER_SEC;
    time_comp += ( clock() - start ) / (double) CLOCKS_PER_SEC;

    if (print_time == 1)
      printf("--- %.2e sec.\n", time_delta_sum);

    /**
    * TEST
    * print out the summed deltas
    */

    // for (int i = 0; i < d; i++)
    // {
    //   torusTemp = LweFunctions<Torus64>::SymDecrypt(lweSummedDeltas[i], out_lwe_key, delta_modulus);
    //   testingArray[i] = TorusUtils<Torus64>::modSwitchFromTorus(torusTemp, delta_modulus);
    // }
    //
    // printf("Summed deltas:\n[");
    // for (int i = 0; i < d-1; i++)
    //   printf("%d, ", testingArray[i]);
    // printf("%d]\n", testingArray[d-1]);

    /**
    * At this point we need to bootstrap every one of them again
    */

    if (print_time == 1)
      std::cout << "- final bootstrap" << '\n';

    // Here we just need the modulus to be 4
    final_modulus = 4;
    positive = TorusUtils<Torus64>::modSwitchToTorus(1, final_modulus);
    negative = TorusUtils<Torus64>::modSwitchToTorus(-1, final_modulus);

    // LweFunctions<Torus64>::NoiselessTrivial(lweInTemp, positive, init_lwe_params);
    // LweFunctions<Torus64>::SymEncrypt(lweInTemp, positive, init_alpha, init_lwe_key);

    start = clock();

    for (int i = 0; i < d; i++)
      TfheFunctions<Torus64>::bootstrap_woKS_FFT(lweArgmax[i], boot_key_fft2, positive, lweSummedDeltas[i]);

    time_final_boot += ( clock() - start ) / (double) CLOCKS_PER_SEC;


    // Rotate them because its aesthetically pleasing

    // Set this one to 1/modulus for rotation pruposes
    LweFunctions<Torus64>::NoiselessTrivial(lweRotation, negative, init_lwe_params);

    for (int i = 0; i < d; i++)
      LweFunctions<Torus64>::AddTo(lweArgmax[i], lweRotation, init_lwe_params);


    time_comp += ( clock() - start ) / (double) CLOCKS_PER_SEC;

    if (print_time == 1)
      printf("--- %.2e sec.\n", time_final_boot);

    /**
    * Overall server time
    */

    if (print_time == 1)
      printf("Overall computation time: %.2e sec.\n", time_comp);

    /**
    * Decryption
    */

    if (print_time == 1)
      std::cout << "- decryption" << '\n';

    start = clock();

    for (int i = 0; i < d; i++)
    {
      torusTemp = LweFunctions<Torus64>::SymDecrypt(lweArgmax[i], init_lwe_key, final_modulus);
      argmax[i] = TorusUtils<Torus64>::modSwitchFromTorus(torusTemp, final_modulus);
    }

    time_decryption += ( clock() - start ) / (double) CLOCKS_PER_SEC;

    if (print_time == 1)
      printf("--- %.2e sec.\n", time_decryption);

    /**
    * Write out the results
    */

    for (int i = 0; i < d-1; i++)
      fprintf(h,"%d,", argmax[i]/2);
    fprintf(h,"%d\n", argmax[d-1]/2);

    /**
    * TEST
    * Print out the results
    */

    // printf("- results:\n[");
    // for (int i = 0; i < d-1; i++)
    //   printf("%d, ", argmax[i]);
    // printf("%d]\n", argmax[d-1]);

    /**
    * Analyse the results
    * - check if there is a single 2 among the results or not
    * -- if not: count as missclassified
    * -- if so: classify the testing vector and check for the actual classification.
    */

    summedResults = 0;
    for (int i = 0; i < d; i++)
      summedResults += argmax[i];

    // If there is a result other than
    if (summedResults == 2)
    {
      for (int i = 0; i < d; i++)
      {
        if (argmax[i] == 2)
        {
          if (true_argmax[i] == 1)
            classification_hits++;
        }
      }
    }
    else
    {
      noresult_occurences++;
    }

    /*
    * Printing the progress here
    */

    nb_loops += 1;
    progress = (float) nb_loops / total_loops;
    std::cout << "[";
    pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();

  }   // End of the big classification loop
  std::cout << std::endl;     // This is for when we have the progress bar over the whole testing thing
  std::cout << '\n';

  /**
  * Print the classification rate
  */

  float percentage_hits = (float) classification_hits*100/number_of_testing_loops;
  float percentage_noresult = (float) noresult_occurences*100/number_of_testing_loops;
  float percentage_hits_when_result;
  if (noresult_occurences != number_of_testing_loops)
    percentage_hits_when_result = (float) classification_hits*100/(number_of_testing_loops-noresult_occurences);
  else
    percentage_hits_when_result = 0;

  std::cout << "Classification rate: ---- " << percentage_hits << " %\n";
  std::cout << "No result rate: --------- " << percentage_noresult << " %\n";
  std::cout << "Rate when result: ------- " << percentage_hits_when_result << " %\n";

  /**
  * Clean up
  */

  // Close the files used for reading
  fclose(f);
  fclose(g);
  fclose(h);

  // The clear data
  for (int i = gamma-1; i >= 0; i--)
    free(labels[i]);

  // free(all_labels);

  // Parameters
  delete init_tgsw_params;
  delete init_tlwe_params;
  delete out_tgsw_params;
  delete out_tlwe_params;

  // Keys
  delete init_tgsw_key;
  delete init_lwe_key;
  delete out_tgsw_key;
  delete out_lwe_key;

  // Bootstrapping keys
  delete boot_key;
  delete boot_key_fft;
  delete boot_key2;
  delete boot_key_fft2;

  // Key-switching key
  // delete key_switch_key;

  // TFHE Samples

  // All of the initial labels
  for (int i = gamma-1; i >= 0; i--)
  {
    for (int j = 0; j < d; j++)
      delete lweLabels[i][j];
    free(lweLabels[i]);
  }

  free(laplacian_noise);

  // Summed labels
  for (int i = d-1; i >= 0; i--)
    delete lweSummedLabels[i];
  free(lweSummedLabels);

  // The amplitude
  delete lweAmplitude;

  // The difference between the labels
  delete lweDifference;

  // The deltas
  for (int i = d-1; i >= 0; i--)
  {
    for (int j = 0; j < d; j++)
      delete lweDeltas[i][j];
    free(lweDeltas[i]);
  }

  // The rotation ciphertext
  delete lweRotation;

  // Summed deltas
  for (int i = d-1; i >= 0; i--)
    delete lweSummedDeltas[i];
  free(lweSummedDeltas);

  // Switched Argmax results
  for (int i = d-1; i >= 0; i--)
    delete lweSwitchedDeltaSum[i];
  free(lweSwitchedDeltaSum);

  // Argmax results
  for (int i = d-1; i >= 0; i--)
    delete lweArgmax[i];
  free(lweArgmax);

  // Clear argmax
  free(argmax);

  free(true_argmax);

  free(summed_labels);

  // Testing array
  free(testingArray);

  // Temporary samples
  delete lweOutTemp;
  delete lweInTemp;

  return 0;
}
