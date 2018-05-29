Implement parallel wavenet based on nsynth.

To make the code and configuration as simple as possible, most of the extensible properties are not extended and are set to default values.

Librosa downsample result may be not in [-1, 1), so use tool/sox_downsample.py to downsample all waves first.


* [OK] wavenet 
* [OK] fastgen for wavenet  
* [OK] parallel wavenet  
* [OK] gen for parallel wavenet

Notice: predicting log_scale from iaf is not a good choice. Because scale = exp(log_scale)  may be too large when running the randomly initialized networks.
This would incur numerical problems in the following  steps.

It seems that using mu law make the training easier. So experiment it first.
* tune wavenet 
    * [OK] use_mu_law + ce ![LJ001-0001](tests/pred_data-use_mu_law+ce/gen_LJ001-0001.wav) ![LJ001-0002](tests/pred_data-use_mu_law+ce/gen_LJ001-0002.wav)
    * [OK] use_mu_law + mol ![LJ001-0001](tests/pred_data-use_mu_law+mol/gen_LJ001-0001.wav) ![LJ001-0002](tests/pred_data-use_mu_law+mol/gen_LJ001-0002.wav)
    * no_mu_law + mol
* tune parallel wavenet 
    * use_mu_law
    * no_mu_law
