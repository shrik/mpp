# install mpp

## install turbojpeg
git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
cd libjpeg-turbo
mkdir build
cd build
<!-- cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -->
cmake ..
make -j8
sudo make install



<!-- sudo apt install libturbojpeg0-dev libturbojpeg -->



git clone git@github.com:shrik/mpp.git mycmpp
cd mycmpp
cd build/linux/aarch64/
bash make-Makefiles.bash
make -j8
make install



# run

./test/custom_enc_test -i /dev/video0 -t 7 -o out.h264 -n 100

python3 extract_frame_info.py out.h264


# crontab

每20分钟从8点到21点执行一次，每次10分钟
*/20 8-21 * * * /usr/local/bin/custom_enc_test -i /dev/video0 -t 7 -o /userdata/tennis/videos/$(date +\%Y\%m\%d_\%H\%M\%S).h264 -n 18000


设置时区TODO

添加自启动frpc
