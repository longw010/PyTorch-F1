# README

Functions included:
- profile::get_inf_time (model, input_size) => get the inference time in GPU mode or CPU mode; note that for the GPU mode
it needs to be in a synchronized way.
- info_str::model, mode, input_size => print out the model strucuture in a tree-based or flat view



Issues to be added or fixed:

info.py
- [ ] trainable parameters setup for info_params in the tree-based layout: need to find the module.weight.requires_grad

prune.py
- [ ] a checklist for one click prune

dryrun.py 
- [ ] use random variables to run profiling.


profile.py
- [ ] add sub func for profile

 