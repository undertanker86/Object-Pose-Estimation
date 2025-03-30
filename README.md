# Test on enviroment
cudatoolkit = 12.6
pytorch = 2.6
# Done Parts:
- [x] Load dataset
- [x] Encoder (resnet 18 or 50)
- [x] Decoder for segmentation
- [x] Decoder for vector-field using voting
- [x] Decoder for depth
- [x] Have code masked_conv.py
- [x] Code run with GPU 
- [x] Integrate Mask conv 
- [x] Teacher-Student
# Need supplement:
- [ ] Cuda kernel for voting because CUDA error: an illegal memory access was encountered
- [ ] Data augmentation
- [ ] More......
## Use config.toml to adjust, and run python.py
