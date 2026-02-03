from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'MOT17-04-SDP-raw.webm')
        cmd_str = 'ffmpeg -loglevel debug -protocol_whitelist pipe,udp,rtp -f sdp -i pipe:0 -map 0:v:0 -c:v libvpx-vp9 -crf 30 -b:v 0 -map 0:a:0 -strict -2 -c:a libvorbis -flags +global_header -f webm {}'.format(output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
