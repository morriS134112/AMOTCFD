import os.path as osp  # 匯入 os.path 模組並將其命名為 osp，方便路徑操作
import os  # 匯入 os 模組，用於操作文件和目錄
import numpy as np  # 匯入 numpy 模組並將其命名為 np，方便數值計算


def mkdirs(d):
    if not osp.exists(d):  # 如果目錄 d 不存在
        os.makedirs(d)  # 創建目錄 d


seq_root = 'E:/FairMOT/dataset/Dance/image/train'  # 設定序列圖像的根目錄
label_root = 'E:/FairMOT/dataset/Dance/labels_with_ids/train'  # 設定標籤的根目錄
mkdirs(label_root)  # 創建標籤根目錄
seqs = [s for s in os.listdir(seq_root)]  # 獲取序列根目錄下的所有文件夾名稱

tid_curr = 0  # 初始化當前的目標 ID
tid_last = -1  # 初始化上一個目標 ID
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()  # 讀取序列信息文件
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])  # 提取圖像寬度
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])  # 提取圖像高度

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')  # 獲取真實值（ground truth）標籤文件的路徑
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')  # 讀取真實值標籤文件

    seq_label_root = osp.join(label_root, seq, 'img1')  # 設定當前序列標籤的存儲目錄
    mkdirs(seq_label_root)  # 創建當前序列標籤的存儲目錄

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:  # 如果該標籤無效或不是行人
            continue  # 跳過這條標籤
        fid = int(fid)  # 轉換幀 ID 為整數
        tid = int(tid)  # 轉換目標 ID 為整數
        if not tid == tid_last:  # 如果當前目標 ID 與上一個不同
            tid_curr += 1  # 當前目標 ID 增加
            tid_last = tid  # 更新上一個目標 ID
        x += w / 2  # 計算目標中心 x 座標
        y += h / 2  # 計算目標中心 y 座標
        label_fpath = osp.join(seq_label_root, '{:08d}.txt'.format(fid))  # 設定當前幀標籤文件的路徑
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)  # 格式化標籤內容
        with open(label_fpath, 'a') as f:  # 打開標籤文件並以追加模式寫入
            f.write(label_str)  # 寫入標籤內容