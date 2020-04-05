# Torch-F1

### Q: What's TorchF1? What is the package for?

F1 is a shortcut for the help. TorchF1 is a simple API to get various info for your PyTorch-based network class.

When you have a new network class in mind, you can apply a dry run on it using the package, and have a basic understanding on the network without deal with all the details on data loader, training details, etc. 

Also, when you plan to do model pruning etc or deploy model in the edge devices, you could use it to test the inference time under different hardware platform without leaking the actual weight info. 

**It is mainly used for personal dev work**

----
## User Guide

### Pre-requisites

- Python 3.6+
- (Recommended) A virtualenv setup for your project

### Features
- [ ] will try to address and fix the potential issues on some popular code snippets
- [ ] dry run in the cpu mode 
- [ ] one line to prune to make the network to be smaller to fit into the GPU memory


