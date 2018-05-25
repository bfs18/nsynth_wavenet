Implement parallel wavenet based on nsynth.

To make the code and configuration as simple as possible, most of the extensible properties are not extended and are set to default values.

Librosa downsample result may be not in [-1, 1), so use tool/sox_downsample.py to downsample all waves first.


* [OK] wavenet 
* [OK] fastgen for wavenet  
* [OK] parallel wavenet  
* [OK] gen for parallel wavenet
* tune wavenet use_mu_law = True
* tune wavenet use_mu_law = False
* tune parallel wavenet use_mu_law = True
* tune parallel wavenet use_mu_law = False
