#define MODULE_TAG "custom_enc_test"

#include <string.h>
#include "rk_mpi.h"

#include "mpp_env.h"
#include "mpp_mem.h"
#include "mpp_time.h"
#include "mpp_debug.h"
#include "mpp_common.h"
#include "mpp_soc.h"

#include "utils.h"
#include "mpi_enc_utils.h"
#include "camera_source.h"
#include "mpp_enc_roi_utils.h"
#include "mpp_rc_api.h"

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <array>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <iomanip>


#define CLEAR(x) memset(&(x), 0, sizeof(x))

std::vector<uint8_t> read_test_image() {
    std::ifstream file("/home/teamhd/Projects/video_distribute/lineman/src/tennis_1920_1080.jpg", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open test image file." << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (file.read(reinterpret_cast<char*>(data.data()), size)) {
        return data;
    } else {
        std::cerr << "Failed to read test image file." << std::endl;
        return {};  
    }
}


class Camera {
public:
    Camera(const std::string& device) : device_(device), fd_(-1), buffer_count_(0) {}
    bool open_camera() {
        fd_ = open(device_.c_str(), O_RDWR);
        if (fd_ == -1) {
            std::cerr << "Error opening device: " << device_ << std::endl;
            return false;
        }

        struct v4l2_capability cap;
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) == -1) {
            std::cerr << "Error querying capabilities." << std::endl;
            return false;
        }

        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            std::cerr << "Device does not support video capture." << std::endl;
            return false;
        }

        struct v4l2_format fmt;
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = 1920;
        fmt.fmt.pix.height = 1080;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
            std::cerr << "Error setting pixel format." << std::endl;
            return false;
        }

        struct v4l2_streamparm parm;
        CLEAR(parm);
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = 120;  // 120 fps, 该相机只支持120fps

        if (ioctl(fd_, VIDIOC_S_PARM, &parm) == -1) {
            perror("Setting Frame Rate");
            return false;
        }

        struct v4l2_requestbuffers req;
        CLEAR(req);
        req.count = 64; // Request 4 buffers
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::cerr << "Error requesting buffers." << std::endl;
            return false;
        }

        buffers_.resize(req.count);

        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf;
            CLEAR(buf);
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) == -1) {
                std::cerr << "Error querying buffer." << std::endl;
                return false;
            }

            buffers_[i].length = buf.length;
            buffers_[i].start = static_cast<uint8_t*>(mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset));

            if (buffers_[i].start == MAP_FAILED) {
                std::cerr << "Error mapping buffer." << std::endl;
                return false;
            }

            if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
                std::cerr << "Error queuing buffer." << std::endl;
                return false;
            }
        }

        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
            std::cerr << "Error starting stream." << std::endl;
            return false;
        }

        return true;
    }

    std::vector<uint8_t> read_frame() {
        struct v4l2_buffer buf;
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
            std::cerr << "Error dequeuing buffer." << std::endl;
            return {};
        }

        std::vector<uint8_t> frame_data(buffers_[buf.index].start, buffers_[buf.index].start + buf.bytesused);

        if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
            std::cerr << "Error queuing buffer." << std::endl;
            return {};
        }

        return frame_data;
    }

    std::pair<std::vector<uint8_t>, unsigned long> read_frame_with_timestamp() {
        auto frame = read_frame();
        // return system milliseconds   
        auto timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000000;
        return std::make_pair(frame, timestamp);
    }

    void stop_capture() {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);

        for (auto& buffer : buffers_) {
            if (buffer.start != nullptr && buffer.start != MAP_FAILED) {
                munmap(buffer.start, buffer.length);
            }
        }

        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
    }

    ~Camera() {
        stop_capture();
    }
    
    bool is_open() const {
        return fd_ != -1;
    }
    
    int get_fd() const {
        return fd_;
    }

private:
    struct Buffer {
        uint8_t* start;
        size_t length;
    };

    std::string device_;
    int fd_;
    std::vector<Buffer> buffers_;
    size_t buffer_count_;
};
// Camera


#include <turbojpeg.h>



std::vector<uint8_t> decode_jpeg(const std::vector<uint8_t>& jpeg_data, int* width, int* height) {
    tjhandle handle = tjInitDecompress();
    if (!handle) {
        std::cerr << "Failed to initialize TurboJPEG decompressor" << std::endl;
        // throw std::runtime_error("Failed to initialize TurboJPEG decompressor");
    }
    
    int jpegSubsamp, jpegColorspace;
    if (tjDecompressHeader3(handle, jpeg_data.data(), jpeg_data.size(), width, height, &jpegSubsamp, &jpegColorspace) < 0) {
        tjDestroy(handle);
        std::cerr << "Failed to read JPEG header" << std::endl;
        // throw std::runtime_error("Failed to read JPEG header");
    }

    std::vector<uint8_t> rgb_data((*width) * (*height) * 3);

    if (tjDecompress2(handle, jpeg_data.data(), jpeg_data.size(), rgb_data.data(), *width, 0, *height, TJPF_RGB, TJFLAG_FASTDCT) < 0) {
        tjDestroy(handle);
        std::cerr << "Failed to decompress JPEG image" << std::endl;
        // throw std::runtime_error("Failed to decompress JPEG image");
    }

    // Convert RGB to YUV420 (NV12)
    // std::vector<uint8_t> yuv_data((*width) * (*height) * 3 / 2);
    // uint8_t *y = yuv_data.data();
    // uint8_t *uv = y + (*width) * (*height);

    // for (int i = 0; i < *height; i++) {
    //     for (int j = 0; j < *width; j++) {
    //         int r = rgb_data[(i * (*width) + j) * 3 + 0];
    //         int g = rgb_data[(i * (*width) + j) * 3 + 1];
    //         int b = rgb_data[(i * (*width) + j) * 3 + 2];

    //         y[i * (*width) + j] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

    //         if (i % 2 == 0 && j % 2 == 0) {
    //             uv[(i / 2) * (*width) + j] = ((-38 * r - 74 * g + 112 * b) >> 8) + 128;
    //             uv[(i / 2) * (*width) + j + 1] = ((112 * r - 94 * g - 18 * b) >> 8) + 128;
    //         }
    //     }
    // }
    // delete[] rgb_data.data();
    tjDestroy(handle);
    return rgb_data;
}

// Usage example:
// std::pair<std::vector<uint8_t>, unsigned long> frame_data = camera.read_frame_with_timestamp();
// int width, height;
// std::vector<uint8_t> rgb_data = decode_jpeg(frame_data.first, &width, &height);
// Now rgb_data contains the decoded image in RGB format





static RK_S32 aq_thd_smart[16] = {
    0,  0,  0,  0,  3,  3,  5,  5,
    8,  8,  8, 15, 15, 20, 25, 28
};

static RK_S32 aq_step_smart[16] = {
    -8, -7, -6, -5, -4, -3, -2, -1,
    0,  1,  2,  3,  4,  6,  8, 10
};

typedef struct {
    // base flow context
    MppCtx ctx;
    MppApi *mpi;
    RK_S32 chn;

    // global flow control flag
    RK_U32 frm_eos;
    RK_U32 pkt_eos;
    RK_U32 frm_pkt_cnt;
    RK_S32 frame_num;
    RK_S32 frame_count;
    RK_U64 stream_size;
    /* end of encoding flag when set quit the loop */
    volatile RK_U32 loop_end;

    // src and dst
    FILE *fp_input;
    FILE *fp_output;
    FILE *fp_verify;

    /* encoder config set */
    MppEncCfg       cfg;
    MppEncPrepCfg   prep_cfg;
    MppEncRcCfg     rc_cfg;
    MppEncCodecCfg  codec_cfg;
    MppEncSliceSplit split_cfg;
    MppEncOSDPltCfg osd_plt_cfg;
    MppEncOSDPlt    osd_plt;
    MppEncOSDData   osd_data;
    RoiRegionCfg    roi_region;
    MppEncROICfg    roi_cfg;

    // input / output
    MppBufferGroup buf_grp;
    MppBuffer frm_buf;
    MppBuffer pkt_buf;
    MppBuffer md_info;
    MppEncSeiMode sei_mode;
    MppEncHeaderMode header_mode;

    // paramter for resource malloc
    RK_U32 width;
    RK_U32 height;
    RK_U32 hor_stride;
    RK_U32 ver_stride;
    MppFrameFormat fmt;
    MppCodingType type;
    RK_S32 loop_times;
    Camera *cam_ctx;
    MppEncRoiCtx roi_ctx;

    // resources
    size_t header_size;
    size_t frame_size;
    size_t mdinfo_size;
    /* NOTE: packet buffer may overflow */
    size_t packet_size;

    RK_U32 osd_enable;
    RK_U32 osd_mode;
    RK_U32 split_mode;
    RK_U32 split_arg;
    RK_U32 split_out;

    RK_U32 user_data_enable;
    RK_U32 roi_enable;

    // rate control runtime parameter
    RK_S32 fps_in_flex;
    RK_S32 fps_in_den;
    RK_S32 fps_in_num;
    RK_S32 fps_out_flex;
    RK_S32 fps_out_den;
    RK_S32 fps_out_num;
    RK_S32 bps;
    RK_S32 bps_max;
    RK_S32 bps_min;
    RK_S32 rc_mode;
    RK_S32 gop_mode;
    RK_S32 gop_len;
    RK_S32 vi_len;
    RK_S32 scene_mode;
    RK_S32 cu_qp_delta_depth;
    RK_S32 anti_flicker_str;
    RK_S32 atr_str_i;
    RK_S32 atr_str_p;
    RK_S32 atl_str;
    RK_S32 sao_str_i;
    RK_S32 sao_str_p;
    RK_S64 first_frm;
    RK_S64 first_pkt;
} MpiEncTestData;

/* For each instance thread return value */
typedef struct {
    float           frame_rate;
    RK_U64          bit_rate;
    RK_S64          elapsed_time;
    RK_S32          frame_count;
    RK_S64          stream_size;
    RK_S64          delay;
} MpiEncMultiCtxRet;

typedef struct {
    MpiEncTestArgs      *cmd;       // pointer to global command line info
    const char          *name;
    RK_S32              chn;

    pthread_t           thd;        // thread for for each instance
    MpiEncTestData      ctx;        // context of encoder
    MpiEncMultiCtxRet   ret;        // return of encoder
    Camera              *camera;
} MpiEncMultiCtxInfo;

static RK_S32 aq_thd[16] = {
    0,  0,  0,  0,
    3,  3,  5,  5,
    8,  8,  8,  15,
    15, 20, 25, 25
};

static RK_S32 aq_step_i_ipc[16] = {
    -8, -7, -6, -5,
    -4, -3, -2, -1,
    0,  1,  2,  3,
    5,  7,  7,  8,
};

static RK_S32 aq_step_p_ipc[16] = {
    -8, -7, -6, -5,
    -4, -2, -1, -1,
    0,  2,  3,  4,
    6,  8,  9,  10,
};

static RK_S32 get_mdinfo_size(MpiEncTestData *p, MppCodingType type)
{
    RockchipSocType soc_type = mpp_get_soc_type();
    RK_S32 md_size;
    RK_U32 w = p->hor_stride, h = p->ver_stride;

    if (soc_type == ROCKCHIP_SOC_RK3588) {
        md_size = (MPP_ALIGN(w, 64) >> 6) * (MPP_ALIGN(h, 64) >> 6) * 32;
    } else {
        md_size = (MPP_VIDEO_CodingHEVC == type) ?
                  (MPP_ALIGN(w, 32) >> 5) * (MPP_ALIGN(h, 32) >> 5) * 16 :
                  (MPP_ALIGN(w, 64) >> 6) * (MPP_ALIGN(h, 16) >> 4) * 16;
    }

    return md_size;
}

MPP_RET test_ctx_init(MpiEncMultiCtxInfo *info)
{
    MpiEncTestArgs *cmd = info->cmd;
    MpiEncTestData *p = &info->ctx;
    MPP_RET ret = MPP_OK;

    // get paramter from cmd
    p->width        = cmd->width;
    p->height       = cmd->height;
    p->hor_stride   = (cmd->hor_stride) ? (cmd->hor_stride) :
                      (MPP_ALIGN(cmd->width, 16));
    p->ver_stride   = (cmd->ver_stride) ? (cmd->ver_stride) :
                      (MPP_ALIGN(cmd->height, 16));
    p->fmt          = cmd->format;
    p->type         = cmd->type;
    p->bps          = cmd->bps_target;
    p->bps_min      = cmd->bps_min;
    p->bps_max      = cmd->bps_max;
    p->rc_mode      = cmd->rc_mode;
    p->frame_num    = cmd->frame_num;
    if (cmd->type == MPP_VIDEO_CodingMJPEG && p->frame_num == 0) {
        std::cout << "jpege default encode only one frame. Use -n [num] for rc case" << std::endl;
        p->frame_num = 1;
    }
    p->gop_mode     = cmd->gop_mode;
    p->gop_len      = cmd->gop_len;
    p->vi_len       = cmd->vi_len;
    p->fps_in_flex  = cmd->fps_in_flex;
    p->fps_in_den   = cmd->fps_in_den;
    p->fps_in_num   = cmd->fps_in_num;
    p->fps_out_flex = cmd->fps_out_flex;
    p->fps_out_den  = cmd->fps_out_den;
    p->fps_out_num  = cmd->fps_out_num;
    p->scene_mode   = cmd->scene_mode;
    p->cu_qp_delta_depth = cmd->cu_qp_delta_depth;
    p->anti_flicker_str = cmd->anti_flicker_str;
    p->atr_str_i = cmd->atr_str_i;
    p->atr_str_p = cmd->atr_str_p;
    p->atl_str = cmd->atl_str;
    p->sao_str_i = cmd->sao_str_i;
    p->sao_str_p = cmd->sao_str_p;
    p->mdinfo_size  = get_mdinfo_size(p, cmd->type);

    if (cmd->file_input) {
        if (!strncmp(cmd->file_input, "/dev/video", 10)) {
            std::cout << "open camera device" << std::endl;
            p->cam_ctx = new Camera(cmd->file_input);
            if (!p->cam_ctx->open_camera()) {
                std::cerr << "failed to open camera " << cmd->file_input << std::endl;
            }
        } else {
            p->fp_input = fopen(cmd->file_input, "rb");
            if (NULL == p->fp_input) {
                std::cerr << "failed to open input file " << cmd->file_input << std::endl;
                std::cerr << "create default yuv image for test" << std::endl;
            }
        }
    }

    if (cmd->file_output) {
        p->fp_output = fopen(cmd->file_output, "w+b");
        if (NULL == p->fp_output) {
            std::cerr << "failed to open output file " << cmd->file_output << std::endl;
            ret = MPP_ERR_OPEN_FILE;
        }
    }

    if (cmd->file_slt) {
        p->fp_verify = fopen(cmd->file_slt, "wt");
        if (!p->fp_verify)
            std::cerr << "failed to open verify file " << cmd->file_slt << std::endl;
    }

    // update resource parameter
    switch (p->fmt & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP:
    case MPP_FMT_YUV420P: {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64) * 3 / 2;
    } break;

    case MPP_FMT_YUV422_YUYV :
    case MPP_FMT_YUV422_YVYU :
    case MPP_FMT_YUV422_UYVY :
    case MPP_FMT_YUV422_VYUY :
    case MPP_FMT_YUV422P :
    case MPP_FMT_YUV422SP : {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64) * 2;
    } break;
    case MPP_FMT_YUV400:
    case MPP_FMT_RGB444 :
    case MPP_FMT_BGR444 :
    case MPP_FMT_RGB555 :
    case MPP_FMT_BGR555 :
    case MPP_FMT_RGB565 :
    case MPP_FMT_BGR565 :
    case MPP_FMT_RGB888 :
    case MPP_FMT_BGR888 :
    case MPP_FMT_RGB101010 :
    case MPP_FMT_BGR101010 :
    case MPP_FMT_ARGB8888 :
    case MPP_FMT_ABGR8888 :
    case MPP_FMT_BGRA8888 :
    case MPP_FMT_RGBA8888 : {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64);
    } break;

    default: {
        p->frame_size = MPP_ALIGN(p->hor_stride, 64) * MPP_ALIGN(p->ver_stride, 64) * 4;
    } break;
    }

    if (MPP_FRAME_FMT_IS_FBC(p->fmt)) {
        if ((p->fmt & MPP_FRAME_FBC_MASK) == MPP_FRAME_FBC_AFBC_V1)
            p->header_size = MPP_ALIGN(MPP_ALIGN(p->width, 16) * MPP_ALIGN(p->height, 16) / 16, SZ_4K);
        else
            p->header_size = MPP_ALIGN(p->width, 16) * MPP_ALIGN(p->height, 16) / 16;
    } else {
        p->header_size = 0;
    }

    return ret;
}

MPP_RET test_ctx_deinit(MpiEncTestData *p)
{
    if (p) {
        if (p->cam_ctx) {
            // camera_source_deinit(p->cam_ctx);
            delete p->cam_ctx;
            p->cam_ctx = NULL;
        }
        if (p->fp_input) {
            fclose(p->fp_input);
            p->fp_input = NULL;
        }
        if (p->fp_output) {
            fclose(p->fp_output);
            p->fp_output = NULL;
        }
        if (p->fp_verify) {
            fclose(p->fp_verify);
            p->fp_verify = NULL;
        }
    }
    return MPP_OK;
}

MPP_RET test_mpp_enc_cfg_setup(MpiEncMultiCtxInfo *info)
{
    MpiEncTestArgs *cmd = info->cmd;
    MpiEncTestData *p = &info->ctx;
    MppApi *mpi = p->mpi;
    MppCtx ctx = p->ctx;
    MppEncCfg cfg = p->cfg;
    RK_U32 quiet = cmd->quiet;
    MPP_RET ret;
    RK_U32 rotation;
    RK_U32 mirroring;
    RK_U32 flip;
    RK_U32 gop_mode = p->gop_mode;
    MppEncRefCfg ref = NULL;
    /* setup default parameter */
    if (p->fps_in_den == 0)
        p->fps_in_den = 1;
    if (p->fps_in_num == 0)
        p->fps_in_num = 30;
    if (p->fps_out_den == 0)
        p->fps_out_den = 1;
    if (p->fps_out_num == 0)
        p->fps_out_num = 30;

    if (!p->bps)
        p->bps = p->width * p->height / 8 * (p->fps_out_num / p->fps_out_den);

    if (cmd->rc_mode == MPP_ENC_RC_MODE_SMTRC) {
        mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_i", aq_thd_smart);
        mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_p", aq_thd_smart);
        mpp_enc_cfg_set_st(cfg, "hw:aq_step_i", aq_step_smart);
        mpp_enc_cfg_set_st(cfg, "hw:aq_step_p", aq_step_smart);
    }

    mpp_enc_cfg_set_s32(cfg, "rc:max_reenc_times", 0);
    mpp_enc_cfg_set_s32(cfg, "rc:cu_qp_delta_depth", p->cu_qp_delta_depth);
    mpp_enc_cfg_set_s32(cfg, "tune:anti_flicker_str", p->anti_flicker_str);
    mpp_enc_cfg_set_s32(cfg, "tune:atr_str_i", p->atr_str_i);
    mpp_enc_cfg_set_s32(cfg, "tune:atr_str_p", p->atr_str_p);
    mpp_enc_cfg_set_s32(cfg, "tune:atl_str", p->atl_str);
    mpp_enc_cfg_set_s32(cfg, "tune:sao_str_i", p->sao_str_i);
    mpp_enc_cfg_set_s32(cfg, "tune:sao_str_p", p->sao_str_p);

    mpp_enc_cfg_set_s32(cfg, "tune:scene_mode", p->scene_mode);
    mpp_enc_cfg_set_s32(cfg, "tune:deblur_en", cmd->deblur_en);
    mpp_enc_cfg_set_s32(cfg, "tune:deblur_str", cmd->deblur_str);
    mpp_enc_cfg_set_s32(cfg, "tune:qpmap_en", 1);
    mpp_enc_cfg_set_s32(cfg, "tune:rc_container", cmd->rc_container);
    mpp_enc_cfg_set_s32(cfg, "tune:vmaf_opt", 0);
    mpp_enc_cfg_set_s32(cfg, "hw:qbias_en", 1);
    mpp_enc_cfg_set_s32(cfg, "hw:qbias_i", cmd->bias_i);
    mpp_enc_cfg_set_s32(cfg, "hw:qbias_p", cmd->bias_p);
    mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_i", aq_thd);
    mpp_enc_cfg_set_st(cfg, "hw:aq_thrd_p", aq_thd);
    mpp_enc_cfg_set_st(cfg, "hw:aq_step_i", aq_step_i_ipc);
    mpp_enc_cfg_set_st(cfg, "hw:aq_step_p", aq_step_p_ipc);
    mpp_enc_cfg_set_s32(cfg, "hw:skip_bias_en", 0);
    mpp_enc_cfg_set_s32(cfg, "hw:skip_bias", 4);
    mpp_enc_cfg_set_s32(cfg, "hw:skip_sad", 8);

    mpp_enc_cfg_set_s32(cfg, "prep:width", p->width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", p->height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", p->hor_stride);
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", p->ver_stride);
    mpp_enc_cfg_set_s32(cfg, "prep:format", p->fmt);
    mpp_enc_cfg_set_s32(cfg, "prep:range", MPP_FRAME_RANGE_JPEG);

    mpp_enc_cfg_set_s32(cfg, "rc:mode", p->rc_mode);
    mpp_enc_cfg_set_u32(cfg, "rc:max_reenc_times", 0);
    mpp_enc_cfg_set_u32(cfg, "rc:super_mode", 0);

    /* fix input / output frame rate */
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", p->fps_in_flex);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", p->fps_in_num);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denom", p->fps_in_den);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", p->fps_out_flex);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", p->fps_out_num);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denom", p->fps_out_den);

    /* drop frame or not when bitrate overflow */
    mpp_enc_cfg_set_u32(cfg, "rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED);
    mpp_enc_cfg_set_u32(cfg, "rc:drop_thd", 20);        /* 20% of max bps */
    mpp_enc_cfg_set_u32(cfg, "rc:drop_gap", 1);         /* Do not continuous drop frame */

    /* setup bitrate for different rc_mode */
    mpp_enc_cfg_set_s32(cfg, "rc:bps_target", p->bps);
    switch (p->rc_mode) {
    case MPP_ENC_RC_MODE_FIXQP : {
        /* do not setup bitrate on FIXQP mode */
    } break;
    case MPP_ENC_RC_MODE_CBR : {
        /* CBR mode has narrow bound */
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", p->bps_max ? p->bps_max : p->bps * 17 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", p->bps_min ? p->bps_min : p->bps * 15 / 16);
    } break;
    case MPP_ENC_RC_MODE_VBR :
    case MPP_ENC_RC_MODE_AVBR : {
        /* VBR mode has wide bound */
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", p->bps_max ? p->bps_max : p->bps * 17 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", p->bps_min ? p->bps_min : p->bps * 1 / 16);
    } break;
    default : {
        /* default use CBR mode */
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", p->bps_max ? p->bps_max : p->bps * 17 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", p->bps_min ? p->bps_min : p->bps * 15 / 16);
    } break;
    }

    /* setup qp for different codec and rc_mode */
    switch (p->type) {
    case MPP_VIDEO_CodingAVC :
    case MPP_VIDEO_CodingHEVC : {
        switch (p->rc_mode) {
        case MPP_ENC_RC_MODE_FIXQP : {
            RK_S32 fix_qp = cmd->qp_init;

            mpp_enc_cfg_set_s32(cfg, "rc:qp_init", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 0);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_i", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_p", fix_qp);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_p", fix_qp);
        } break;
        case MPP_ENC_RC_MODE_CBR :
        case MPP_ENC_RC_MODE_VBR :
        case MPP_ENC_RC_MODE_AVBR :
        case MPP_ENC_RC_MODE_SMTRC : {
            mpp_enc_cfg_set_s32(cfg, "rc:qp_init", cmd->qp_init ? cmd->qp_init : -1);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max", cmd->qp_max ? cmd->qp_max : 51);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min", cmd->qp_min ? cmd->qp_min : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", cmd->qp_max_i ? cmd->qp_max_i : 51);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", cmd->qp_min_i ? cmd->qp_min_i : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 2);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_i", cmd->fqp_min_i ? cmd->fqp_min_i : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_i", cmd->fqp_max_i ? cmd->fqp_max_i : 45);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_min_p", cmd->fqp_min_p ? cmd->fqp_min_p : 10);
            mpp_enc_cfg_set_s32(cfg, "rc:fqp_max_p", cmd->fqp_max_p ? cmd->fqp_max_p : 45);
        } break;
        default : {
            std::cerr << "unsupport encoder rc mode " << p->rc_mode << std::endl;
        } break;
        }
    } break;
    case MPP_VIDEO_CodingVP8 : {
        /* vp8 only setup base qp range */
        mpp_enc_cfg_set_s32(cfg, "rc:qp_init", cmd->qp_init ? cmd->qp_init : 40);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_max",  cmd->qp_max ? cmd->qp_max : 127);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_min",  cmd->qp_min ? cmd->qp_min : 0);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", cmd->qp_max_i ? cmd->qp_max_i : 127);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", cmd->qp_min_i ? cmd->qp_min_i : 0);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 6);
    } break;
    case MPP_VIDEO_CodingMJPEG : {
        /* jpeg use special codec config to control qtable */
        mpp_enc_cfg_set_s32(cfg, "jpeg:q_factor", cmd->qp_init ? cmd->qp_init : 80);
        mpp_enc_cfg_set_s32(cfg, "jpeg:qf_max", cmd->qp_max ? cmd->qp_max : 99);
        mpp_enc_cfg_set_s32(cfg, "jpeg:qf_min", cmd->qp_min ? cmd->qp_min : 1);
    } break;
    default : {
    } break;
    }

    /* setup codec  */
    mpp_enc_cfg_set_s32(cfg, "codec:type", p->type);
    switch (p->type) {
    case MPP_VIDEO_CodingAVC : {
        RK_U32 constraint_set;

        /*
         * H.264 profile_idc parameter
         * 66  - Baseline profile
         * 77  - Main profile
         * 100 - High profile
         */
        mpp_enc_cfg_set_s32(cfg, "h264:profile", 100);
        /*
         * H.264 level_idc parameter
         * 10 / 11 / 12 / 13    - qcif@15fps / cif@7.5fps / cif@15fps / cif@30fps
         * 20 / 21 / 22         - cif@30fps / half-D1@@25fps / D1@12.5fps
         * 30 / 31 / 32         - D1@25fps / 720p@30fps / 720p@60fps
         * 40 / 41 / 42         - 1080p@30fps / 1080p@30fps / 1080p@60fps
         * 50 / 51 / 52         - 4K@30fps
         */
        mpp_enc_cfg_set_s32(cfg, "h264:level", 40);
        mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", 1);
        mpp_enc_cfg_set_s32(cfg, "h264:cabac_idc", 0);
        mpp_enc_cfg_set_s32(cfg, "h264:trans8x8", 1);

        mpp_env_get_u32("constraint_set", &constraint_set, 0);
        if (constraint_set & 0x3f0000)
            mpp_enc_cfg_set_s32(cfg, "h264:constraint_set", constraint_set);
    } break;
    case MPP_VIDEO_CodingHEVC :
    case MPP_VIDEO_CodingMJPEG :
    case MPP_VIDEO_CodingVP8 : {
    } break;
    default : {
        std::cerr << "unsupport encoder coding type " << p->type << std::endl;
    } break;
    }

    p->split_mode = 0;
    p->split_arg = 0;
    p->split_out = 0;

    mpp_env_get_u32("split_mode", &p->split_mode, MPP_ENC_SPLIT_NONE);
    mpp_env_get_u32("split_arg", &p->split_arg, 0);
    mpp_env_get_u32("split_out", &p->split_out, 0);

    if (p->split_mode) {
        std::cout << "split mode " << p->split_mode << " arg " << p->split_arg << " out " << p->split_out << std::endl;
        mpp_enc_cfg_set_s32(cfg, "split:mode", p->split_mode);
        mpp_enc_cfg_set_s32(cfg, "split:arg", p->split_arg);
        mpp_enc_cfg_set_s32(cfg, "split:out", p->split_out);
    }

    mpp_env_get_u32("mirroring", &mirroring, 0);
    mpp_env_get_u32("rotation", &rotation, 0);
    mpp_env_get_u32("flip", &flip, 0);

    mpp_enc_cfg_set_s32(cfg, "prep:mirroring", mirroring);
    mpp_enc_cfg_set_s32(cfg, "prep:rotation", rotation);
    mpp_enc_cfg_set_s32(cfg, "prep:flip", flip);

    // config gop_len and ref cfg
    mpp_enc_cfg_set_s32(cfg, "rc:gop", p->gop_len ? p->gop_len : p->fps_out_num * 2);

    mpp_env_get_u32("gop_mode", &gop_mode, gop_mode);
    if (gop_mode) {
        mpp_enc_ref_cfg_init(&ref);

        if (p->gop_mode < 4)
            mpi_enc_gen_ref_cfg(ref, gop_mode);
        else
            mpi_enc_gen_smart_gop_ref_cfg(ref, p->gop_len, p->vi_len);

        mpp_enc_cfg_set_ptr(cfg, "rc:ref_cfg", ref);
    }

    ret = mpi->control(ctx, MPP_ENC_SET_CFG, cfg);
    if (ret) {
        std::cerr << "mpi control enc set cfg failed ret " << ret << std::endl;
        goto RET;
    }

    if (cmd->type == MPP_VIDEO_CodingAVC || cmd->type == MPP_VIDEO_CodingHEVC) {
        RcApiBrief rc_api_brief;
        rc_api_brief.type = cmd->type;
        rc_api_brief.name = (cmd->rc_mode == MPP_ENC_RC_MODE_SMTRC) ?
                            "smart" : "default";

        ret = mpi->control(ctx, MPP_ENC_SET_RC_API_CURRENT, &rc_api_brief);
        if (ret) {
            std::cerr << "mpi control enc set rc api failed ret " << ret << std::endl;
            goto RET;
        }
    }

    if (ref)
        mpp_enc_ref_cfg_deinit(&ref);

    /* optional */
    {
        RK_U32 sei_mode;

        mpp_env_get_u32("sei_mode", &sei_mode, MPP_ENC_SEI_MODE_ONE_FRAME);
        p->sei_mode = (MppEncSeiMode)sei_mode;
        ret = mpi->control(ctx, MPP_ENC_SET_SEI_CFG, &p->sei_mode);
        if (ret) {
            std::cerr << "mpi control enc set sei cfg failed ret " << ret << std::endl;
            goto RET;
        }
    }

    if (p->type == MPP_VIDEO_CodingAVC || p->type == MPP_VIDEO_CodingHEVC) {
        p->header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        ret = mpi->control(ctx, MPP_ENC_SET_HEADER_MODE, &p->header_mode);
        if (ret) {
            std::cerr << "mpi control enc set header mode failed ret " << ret << std::endl;
            goto RET;
        }
    }

    /* setup test mode by env */
    mpp_env_get_u32("osd_enable", &p->osd_enable, 0);
    mpp_env_get_u32("osd_mode", &p->osd_mode, MPP_ENC_OSD_PLT_TYPE_DEFAULT);
    mpp_env_get_u32("roi_enable", &p->roi_enable, 0);
    mpp_env_get_u32("user_data_enable", &p->user_data_enable, 1);

    if (p->roi_enable) {
        mpp_enc_roi_init(&p->roi_ctx, p->width, p->height, p->type, 4);
        mpp_assert(p->roi_ctx);
    }

RET:
    return ret;
}

MPP_RET test_mpp_run(MpiEncMultiCtxInfo *info)
{
    MpiEncTestArgs *cmd = info->cmd;
    MpiEncTestData *p = &info->ctx;
    MppApi *mpi = p->mpi;
    MppCtx ctx = p->ctx;
    RK_U32 quiet = cmd->quiet;
    RK_S32 chn = info->chn;
    // RK_U32 cap_num = 0;
    DataCrc checkcrc;
    MPP_RET ret = MPP_OK;
    memset(&checkcrc, 0, sizeof(checkcrc));
    checkcrc.sum = mpp_malloc(RK_ULONG, 512);

    if (p->type == MPP_VIDEO_CodingAVC || p->type == MPP_VIDEO_CodingHEVC) {
        MppPacket packet = NULL;

        /*
         * Can use packet with normal malloc buffer as input not pkt_buf.
         * Please refer to vpu_api_legacy.cpp for normal buffer case.
         * Using pkt_buf buffer here is just for simplifing demo.
         */
        mpp_packet_init_with_buffer(&packet, p->pkt_buf);
        /* NOTE: It is important to clear output packet length!! */
        mpp_packet_set_length(packet, 0);

        ret = mpi->control(ctx, MPP_ENC_GET_HDR_SYNC, packet);
        if (ret) {
            std::cerr << "mpi control enc get extra info failed" << std::endl;
            goto RET;
        } else {
            /* get and write sps/pps for H.264 */

            void *ptr   = mpp_packet_get_pos(packet);
            size_t len  = mpp_packet_get_length(packet);

            if (p->fp_output)
                fwrite(ptr, 1, len, p->fp_output);
        }

        mpp_packet_deinit(&packet);
    }
    while (!p->pkt_eos) {
        MppMeta meta = NULL;
        MppFrame frame = NULL;
        MppPacket packet = NULL;
        void *buf = mpp_buffer_get_ptr(p->frm_buf);
        // RK_S32 cam_frm_idx = -1;
        // MppBuffer cam_buf = NULL;
        RK_U32 eoi = 1;

        mpp_buffer_sync_begin(p->frm_buf);

        p->cam_ctx->read_frame_with_timestamp();
        p->cam_ctx->read_frame_with_timestamp();
        p->cam_ctx->read_frame_with_timestamp();

        auto [camFrame, timestamp] = p->cam_ctx->read_frame_with_timestamp();
        int width = (int) p->width;
        int height = (int) p->height;
        auto res =  decode_jpeg(camFrame, (int *) &width, (int *) &height);

        // auto res = decode_jpeg(read_test_image(), (int *) &p->width, (int *) &p->height);

        // std::cout << "width " << width << " height " << height<< " data size " << res.size() << std::endl;
        memcpy(buf, res.data(), res.size());
        
        mpp_buffer_sync_end(p->frm_buf);
        

        ret = mpp_frame_init(&frame);
        if (ret) {
            std::cerr << "mpp_frame_init failed" << std::endl;
            goto RET;
        }
        std::cout << "frame width " << p->width << " height " << p->height << " fmt " << p->fmt << " frm_eos " << p->frm_eos << std::endl;
        mpp_frame_set_width(frame, p->width);
        mpp_frame_set_height(frame, p->height);
        mpp_frame_set_hor_stride(frame, p->width);
        mpp_frame_set_ver_stride(frame, p->height);
        mpp_frame_set_fmt(frame, p->fmt);
        mpp_frame_set_eos(frame, p->frm_eos);

        mpp_frame_set_buffer(frame, p->frm_buf);

        meta = mpp_frame_get_meta(frame);
        mpp_packet_init_with_buffer(&packet, p->pkt_buf);
        /* NOTE: It is important to clear output packet length!! */
        mpp_packet_set_length(packet, 0);
        mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);
        mpp_meta_set_buffer(meta, KEY_MOTION_INFO, p->md_info);


// ------------------------------------------------------------------------------------------------
        MppEncUserData user_data;
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S.")
           << std::setfill('0') << std::setw(3)
           << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
        std::string str = ss.str();
        user_data.pdata = const_cast<void*>(static_cast<const void*>(str.c_str()));
        user_data.len = str.length() + 1;

        mpp_meta_set_ptr(meta, KEY_USER_DATA, &user_data);
        mpp_meta_set_s32(meta, KEY_TEMPORAL_ID,1);

        std::cout << "frame has meta: " << mpp_frame_has_meta(frame) << std::endl;
// ------------------------------------------------------------------------------------------------
        // MppEncUserData user_data;
        // std::string str = "this is user data\n";

        
        // user_data.pdata = const_cast<void*>(static_cast<const void*>(str.c_str()));
        // user_data.len = str.length() + 1;
        // mpp_meta_set_ptr(meta, KEY_USER_DATA, &user_data);

        // static RK_U8 uuid_debug_info[16] = {
        //     0x57, 0x68, 0x97, 0x80, 0xe7, 0x0c, 0x4b, 0x65,
        //     0xa9, 0x06, 0xae, 0x29, 0x94, 0x11, 0xcd, 0x9a
        // };

        // MppEncUserDataSet data_group;
        // MppEncUserDataFull datas[2];
        // std::string str1 = "this is user data 1\n";
        // std::string str2 = "this is user data 2\n";
        // data_group.count = 2;
        // datas[0].len = str1.length() + 1;
        // datas[0].pdata = const_cast<void*>(static_cast<const void*>(str1.c_str()));
        // datas[0].uuid = uuid_debug_info;

        // datas[1].len = str2.length() + 1;
        // datas[1].pdata = const_cast<void*>(static_cast<const void*>(str2.c_str()));
        // datas[1].uuid = uuid_debug_info;

        // data_group.datas = datas;

        // mpp_meta_set_ptr(meta, KEY_USER_DATAS, &data_group);
        // std::cout << "p->user_data_enable: set user data" << std::endl;

// ------------------------------------------------------------------------------------------------

        if (!p->first_frm)
            p->first_frm = mpp_time();
        /*
         * NOTE: in non-block mode the frame can be resent.
         * The default input timeout mode is block.
         *
         * User should release the input frame to meet the requirements of
         * resource creator must be the resource destroyer.
         */
        ret = mpi->encode_put_frame(ctx, frame);
        if (ret) {
            std::cerr << "chn " << chn << " encode put frame failed" << std::endl;
            mpp_frame_deinit(&frame);
            goto RET;
        }

        mpp_frame_deinit(&frame);

        do {
            ret = mpi->encode_get_packet(ctx, &packet);
            if (ret) {
                std::cerr << "chn " << chn << " encode get packet failed" << std::endl;
                goto RET;
            }

            mpp_assert(packet);

            if (packet) {
                // write packet to file here
                void *ptr   = mpp_packet_get_pos(packet);
                size_t len  = mpp_packet_get_length(packet);
                char log_buf[256];
                RK_S32 log_size = sizeof(log_buf) - 1;
                RK_S32 log_len = 0;

                if (!p->first_pkt)
                    p->first_pkt = mpp_time();

                p->pkt_eos = mpp_packet_get_eos(packet);

                if (p->fp_output)
                    fwrite(ptr, 1, len, p->fp_output);

                if (p->fp_verify && !p->pkt_eos) {
                    calc_data_crc((RK_U8 *)ptr, (RK_U32)len, &checkcrc);
                    std::cout << "p->frame_count=" << p->frame_count << ", len=" << len << std::endl;
                    write_data_crc(p->fp_verify, &checkcrc);
                }

                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                    "encoded frame %-4d", p->frame_count);

                /* for low delay partition encoding */
                if (mpp_packet_is_partition(packet)) {
                    eoi = mpp_packet_is_eoi(packet);

                    log_len += snprintf(log_buf + log_len, log_size - log_len,
                                        " pkt %d", p->frm_pkt_cnt);
                    p->frm_pkt_cnt = (eoi) ? (0) : (p->frm_pkt_cnt + 1);
                }

                log_len += snprintf(log_buf + log_len, log_size - log_len,
                                    " size %-7zu", len);

                if (mpp_packet_has_meta(packet)) {
                    meta = mpp_packet_get_meta(packet);
                    RK_S32 temporal_id = 0;
                    RK_S32 lt_idx = -1;
                    RK_S32 avg_qp = -1, bps_rt = -1;
                    RK_S32 use_lt_idx = -1;

                    mpp_meta_get_s32(meta, KEY_TEMPORAL_ID, &temporal_id);
                    std::cout << "temporal_id: " << temporal_id << std::endl;

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_TEMPORAL_ID, &temporal_id))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " tid %d", temporal_id);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_LONG_REF_IDX, &lt_idx))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " lt %d", lt_idx);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_AVERAGE_QP, &avg_qp))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " qp %2d", avg_qp);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_BPS_RT, &bps_rt))
                        log_len += snprintf(log_buf + log_len, log_size - log_len,
                                            " bps_rt %d", bps_rt);

                    if (MPP_OK == mpp_meta_get_s32(meta, KEY_ENC_USE_LTR, &use_lt_idx))
                        log_len += snprintf(log_buf + log_len, log_size - log_len, " vi");
                }else{
                    std::cout << "p->user_data_enable: no meta" << std::endl;
                }
                std::cout << "chn " << chn << " " << log_buf << std::endl;

                mpp_packet_deinit(&packet);
                fps_calc_inc(cmd->fps);

                p->stream_size += len;
                p->frame_count += eoi;

                if (p->pkt_eos) {
                    std::cout << "chn " << chn << " found last packet" << std::endl;
                    mpp_assert(p->frm_eos);
                }
            }
        } while (!eoi);

        // if (cam_frm_idx >= 0)
        //     camera_source_put_frame(p->cam_ctx, cam_frm_idx);

        if (p->frame_num > 0 && p->frame_count >= p->frame_num)
            break;

        if (p->loop_end)
            break;

        if (p->frm_eos && p->pkt_eos)
            break;
    }
RET:
    MPP_FREE(checkcrc.sum);

    return ret;
}

void *enc_test(void *arg)
{
    MpiEncMultiCtxInfo *info = (MpiEncMultiCtxInfo *)arg;
    MpiEncTestArgs *cmd = info->cmd;
    MpiEncTestData *p = &info->ctx;
    MpiEncMultiCtxRet *enc_ret = &info->ret;
    MppPollType timeout = MPP_POLL_BLOCK;
    RK_U32 quiet = cmd->quiet;
    MPP_RET ret = MPP_OK;
    RK_S64 t_s = 0;
    RK_S64 t_e = 0;

    std::cout << info->name << " start" << std::endl;

    ret = test_ctx_init(info);
    if (ret) {
        std::cout << "test data init failed ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = mpp_buffer_group_get_internal(&p->buf_grp, MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CACHABLE);
    if (ret) {
        std::cout << "failed to get mpp buffer group ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = mpp_buffer_get(p->buf_grp, &p->frm_buf, p->frame_size + p->header_size);
    if (ret) {
        std::cout << "failed to get buffer for input frame ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = mpp_buffer_get(p->buf_grp, &p->pkt_buf, p->frame_size);
    if (ret) {
        std::cout << "failed to get buffer for output packet ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = mpp_buffer_get(p->buf_grp, &p->md_info, p->mdinfo_size);
    if (ret) {
        std::cout << "failed to get buffer for motion info output packet ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    // encoder demo
    ret = mpp_create(&p->ctx, &p->mpi);
    if (ret) {
        std::cout << "mpp_create failed ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    std::cout << "encoder test start w " << p->width << " h " << p->height << " type " << p->type << std::endl;


    ret = p->mpi->control(p->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout);
    if (MPP_OK != ret) {
        std::cout << "mpi control set output timeout " << timeout << " ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = mpp_init(p->ctx, MPP_CTX_ENC, p->type);
    if (ret) {
        std::cout << "mpp_init failed ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = mpp_enc_cfg_init(&p->cfg);
    if (ret) {
        std::cout << "mpp_enc_cfg_init failed ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = p->mpi->control(p->ctx, MPP_ENC_GET_CFG, p->cfg);
    if (ret) {
        std::cout << "get enc cfg failed ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = test_mpp_enc_cfg_setup(info);
    if (ret) {
        std::cout << "test mpp setup failed ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    t_s = mpp_time();
    ret = test_mpp_run(info);
    t_e = mpp_time();
    if (ret) {
        std::cout << "test mpp run failed ret " << ret << std::endl;
        goto MPP_TEST_OUT;
    }

    ret = p->mpi->reset(p->ctx);
    if (ret) {
        std::cout << "mpi->reset failed" << std::endl;
        goto MPP_TEST_OUT;
    }

    enc_ret->elapsed_time = t_e - t_s;
    enc_ret->frame_count = p->frame_count;
    enc_ret->stream_size = p->stream_size;
    enc_ret->frame_rate = (float)p->frame_count * 1000000 / enc_ret->elapsed_time;
    enc_ret->bit_rate = (p->stream_size * 8 * (p->fps_out_num / p->fps_out_den)) / p->frame_count;
    enc_ret->delay = p->first_pkt - p->first_frm;

MPP_TEST_OUT:
    if (p->ctx) {
        mpp_destroy(p->ctx);
        p->ctx = NULL;
    }

    if (p->cfg) {
        mpp_enc_cfg_deinit(p->cfg);
        p->cfg = NULL;
    }

    if (p->frm_buf) {
        mpp_buffer_put(p->frm_buf);
        p->frm_buf = NULL;
    }

    if (p->pkt_buf) {
        mpp_buffer_put(p->pkt_buf);
        p->pkt_buf = NULL;
    }

    if (p->md_info) {
        mpp_buffer_put(p->md_info);
        p->md_info = NULL;
    }

    if (p->osd_data.buf) {
        mpp_buffer_put(p->osd_data.buf);
        p->osd_data.buf = NULL;
    }

    if (p->buf_grp) {
        mpp_buffer_group_put(p->buf_grp);
        p->buf_grp = NULL;
    }

    if (p->roi_ctx) {
        mpp_enc_roi_deinit(p->roi_ctx);
        p->roi_ctx = NULL;
    }

    test_ctx_deinit(p);

    return NULL;
}

int enc_test_multi(MpiEncTestArgs* cmd, const char *name)
{
    MpiEncMultiCtxInfo *ctxs = NULL;
    float total_rate = 0.0;
    RK_S32 ret = MPP_NOK;
    RK_S32 i = 0;

    ctxs = mpp_calloc(MpiEncMultiCtxInfo, cmd->nthreads);
    if (NULL == ctxs) {
        std::cerr << "failed to alloc context for instances" << std::endl;
        return -1;
    }

    for (i = 0; i < cmd->nthreads; i++) {
        ctxs[i].cmd = cmd;
        ctxs[i].name = name;
        ctxs[i].chn = i;

        ret = pthread_create(&ctxs[i].thd, NULL, enc_test, &ctxs[i]);
        if (ret) {
            std::cerr << "failed to create thread " << i << std::endl;
            return ret;
        }
    }

    if (cmd->frame_num < 0) {
        // wait for input then quit encoding
        std::cout << "*******************************************" << std::endl;
        std::cout << "**** Press Enter to stop loop encoding ****" << std::endl;
        std::cout << "*******************************************" << std::endl;

        getc(stdin);
        for (i = 0; i < cmd->nthreads; i++)
            ctxs[i].ctx.loop_end = 1;
    }

    for (i = 0; i < cmd->nthreads; i++)
        pthread_join(ctxs[i].thd, NULL);

    for (i = 0; i < cmd->nthreads; i++) {
        MpiEncMultiCtxRet *enc_ret = &ctxs[i].ret;

        std::cout << "chn " << i << " encode " << enc_ret->frame_count << " frames time " 
                  << (RK_S64)(enc_ret->elapsed_time / 1000) << " ms delay " 
                  << (RK_S32)(enc_ret->delay / 1000) << " ms fps " << enc_ret->frame_rate 
                  << " bps " << enc_ret->bit_rate << std::endl;

        total_rate += enc_ret->frame_rate;
    }

    MPP_FREE(ctxs);

    total_rate /= cmd->nthreads;
    std::cout << name << " average frame rate " << total_rate << std::endl;
    std::cout << "total_rate: " << total_rate << " name: " << name << std::endl;
    return ret;
}




int main(int argc, char **argv)
{
    RK_S32 ret = MPP_NOK;
    MpiEncTestArgs* cmd = mpi_enc_test_cmd_get();

    cmd->fps_out_num = 30;
    // cmd->format = MPP_FMT_YUV420SP;
    cmd->format = MPP_FMT_RGB888;
    cmd->width = 1920;
    cmd->height = 1080;
    cmd->nthreads = 1;
    cmd->frame_num = 300;
    cmd->file_input = const_cast<char*>("/dev/video0");
    cmd->file_output = const_cast<char*>("output.h264");

    std::cout << "argc: " << argc << std::endl;
    // parse the cmd option
    ret = mpi_enc_test_cmd_update_by_args(cmd, argc, argv);
    std::cout << "ret: " << ret << std::endl;

    if (ret)
        goto DONE;

    std::cout << "not goto DONE "  << std::endl;
    mpi_enc_test_cmd_show_opt(cmd);

    std::cout << "mpi_enc_test_cmd_show_opt: " << std::endl;
    ret = enc_test_multi(cmd, argv[0]);
    std::cout << "enc_test_multi: " << ret << std::endl;

DONE:
    mpi_enc_test_cmd_put(cmd);

    return ret;
}
