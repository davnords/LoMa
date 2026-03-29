mkdir -p data/megadepth/
mkdir -p data/scannet/scans/
wget https://github.com/Parskatt/storage/releases/download/mega1500/0015_0.1_0.3.npz && mv 0015_0.1_0.3.npz data/megadepth/
wget https://github.com/Parskatt/storage/releases/download/mega1500/0015_0.3_0.5.npz && mv 0015_0.3_0.5.npz data/megadepth/
wget https://github.com/Parskatt/storage/releases/download/mega1500/0022_0.1_0.3.npz && mv 0022_0.1_0.3.npz data/megadepth/
wget https://github.com/Parskatt/storage/releases/download/mega1500/0022_0.3_0.5.npz && mv 0022_0.3_0.5.npz data/megadepth/
wget https://github.com/Parskatt/storage/releases/download/mega1500/0022_0.5_0.7.npz && mv 0022_0.5_0.7.npz data/megadepth/
wget https://github.com/Parskatt/storage/releases/download/mega1500/megadepth_test_1500.tar && tar -xvf megadepth_test_1500.tar && mv megadepth_test_1500/Undistorted_SfM data/megadepth/ && rmdir megadepth_test_1500 && rm megadepth_test_1500.tar

wget https://github.com/Parskatt/storage/releases/download/scannet1500/test.npz && mv test.npz data/scannet/scans/
wget https://github.com/Parskatt/storage/releases/download/scannet1500/scannet_test_1500.tar && tar -xvf scannet_test_1500.tar && mv scannet_test_1500 data/scannet/scans/ && mv data/scannet/scans/scannet_test_1500 data/scannet/scans/scans_test && rm scannet_test_1500.tar