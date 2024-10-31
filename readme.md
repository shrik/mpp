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



# 设置时区
 date +%Z # check timezone
 sudo timedatectl set-timezone Asia/Shanghai
 date +%Z # check timezone

# 添加自启动frpc
cd opt
wget https://github.com/fatedier/frp/releases/download/v0.60.0/frp_0.60.0_linux_arm64.tar.gz
tar -zxvf frp_0.60.0_linux_arm64.tar.gz
cd frp_0.60.0_linux_arm64
sudo cp frpc /usr/local/bin/
#########
[Unit]
Description=frpc
After=network.target

[Service]
ExecStart=/usr/bin/frpc -c /etc/frp/frpc.toml
Restart=always
RestartSec=60s
User=root

[Install]
WantedBy=multi-user.target
#########

sudo cp frpc.service /etc/systemd/system/

#########
serverAddr = "39.106.81.243"
serverPort = 2222

[[proxies]]
name = "rk3588s"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 6001

#########
sudo mkdir -p /etc/frp
sudo cp frpc.toml /etc/frp/

sudo systemctl enable frpc
sudo systemctl start frpc



# 设置NTP统一时间
sudo apt install ntp
/etc/ntp.conf
pool ntp.aliyun.com iburst
