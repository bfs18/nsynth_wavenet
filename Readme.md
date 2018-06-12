Implement parallel wavenet based on nsynth.

To make the code and configuration as simple as possible, most of the extensible properties are not extended and are set to default values.

Librosa downsample result may be not in [-1, 1), so use tool/sox_downsample.py to downsample all waves first.


* [OK] wavenet 
* [OK] fastgen for wavenet  
* [OK] parallel wavenet  
* [OK] gen for parallel wavenet


It seems that using mu law make the training easier. So experiment it first.  
The following examples are more of functional test than gaining good waves. The network may be not trained enough.
* tune wavenet 
    * [OK] use_mu_law + ce [LJ001-0001](tests/pred_data-use_mu_law+ce/gen_LJ001-0001.wav) [LJ001-0002](tests/pred_data-use_mu_law+ce/gen_LJ001-0002.wav)
    * [OK] use_mu_law + mol [LJ001-0001](tests/pred_data-use_mu_law+mol/gen_LJ001-0001.wav) [J001-0002](tests/pred_data-use_mu_law+mol/gen_LJ001-0002.wav)
    * [OK] no_mu_law + mol [LJ001-0001](tests/pred_data-no_mu_law+mol/gen_LJ001-0001.wav) [LJ001-0002](tests/pred_data-no_mu_law+mol/gen_LJ001-0002.wav)
* tune parallel wavenet 
    * use_mu_law
    * no_mu_law


Proper initial mean_tot and scale_tot values have positive impact on model convergence and numerical stability.
According to the LJSpeech data distribution, proper initial values for mean_tot  and scale_tot should be 0.0 and 0.05.
I modified the initializer to achieve it.  
![data dist](tests/dist2.png)   
The figure is pot by [this script](tests/test_wave_distribution.py)