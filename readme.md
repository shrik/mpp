# install mpp

apt install libturbojpeg-dev

cd build/linux/aarch64/
bash make-Makefiles.bash
make -j8
make install



# run

./test/custom_enc_test -i /dev/video0 -t 7 -o out.h264 -n 100

python3 extract_frame_info.py out.h264