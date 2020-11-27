# TPolicies
Policies used in various RL or IL applications.
Based on tf 1.x APIs.
In the style of `tf.contrib.layers` and `tf-slim`.
Some code of implementing policy gradient loss is borrowed from [`deepmind/trfl`](https://github.com/deepmind/trfl) and [`openai/baselines`](https://github.com/openai/baselines).
Some code for transformer layer is borrowed from [here](https://github.com/Kyubyong/transformer).

## Install
cd to the folder and run the command:
```
pip install -e .
```
Moreover,
we require `tensorflow==1.15.0`, please install it manually!

## Quick Example
See the testing file `net_zoo/net_name/net_name_test.py` which can serve as examples.

## some notes
Use `tf.contrib.framework.nest`, e.g., `nest.flatten`, `nest.map_structure`.
It is helpful when, for example, computing losses from all the action heads.

# Disclaimer
This is not an officially supported Tencent product.
The code and data in this repository are for research purpose only. 
No representation or warranty whatsoever, expressed or implied, is made as to its accuracy, reliability or completeness. 
We assume no liability and are not responsible for any misuse or damage caused by the code and data. 
Your use of the code and data are subject to applicable laws and your use of them is at your own risk.

