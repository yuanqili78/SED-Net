import os
import numpy as np
import glob

fns = glob.glob("C:/Users/Sean/Desktop/edge_log/gt_1000/gt/*_type.txt")

for fn in fns:
    tmp = np.loadtxt(fn)
    sz = tmp[tmp == 0].shape[0]
    if sz > 0:
        print(fn, sz)

"""GT 500
C:/Users/Sean/Desktop/edge_log/gt_500/gt\102_type.txt 1126
C:/Users/Sean/Desktop/edge_log/gt_500/gt\130_type.txt 1880
C:/Users/Sean/Desktop/edge_log/gt_500/gt\135_type.txt 448
C:/Users/Sean/Desktop/edge_log/gt_500/gt\144_type.txt 9
C:/Users/Sean/Desktop/edge_log/gt_500/gt\169_type.txt 2173
C:/Users/Sean/Desktop/edge_log/gt_500/gt\172_type.txt 3239
C:/Users/Sean/Desktop/edge_log/gt_500/gt\178_type.txt 793
C:/Users/Sean/Desktop/edge_log/gt_500/gt\179_type.txt 664
C:/Users/Sean/Desktop/edge_log/gt_500/gt\188_type.txt 169
C:/Users/Sean/Desktop/edge_log/gt_500/gt\21_type.txt 246
C:/Users/Sean/Desktop/edge_log/gt_500/gt\226_type.txt 1860
C:/Users/Sean/Desktop/edge_log/gt_500/gt\268_type.txt 7114
C:/Users/Sean/Desktop/edge_log/gt_500/gt\297_type.txt 843
C:/Users/Sean/Desktop/edge_log/gt_500/gt\300_type.txt 53
C:/Users/Sean/Desktop/edge_log/gt_500/gt\33_type.txt 605
C:/Users/Sean/Desktop/edge_log/gt_500/gt\357_type.txt 443
C:/Users/Sean/Desktop/edge_log/gt_500/gt\36_type.txt 445
C:/Users/Sean/Desktop/edge_log/gt_500/gt\376_type.txt 181
C:/Users/Sean/Desktop/edge_log/gt_500/gt\396_type.txt 2570
C:/Users/Sean/Desktop/edge_log/gt_500/gt\403_type.txt 194
C:/Users/Sean/Desktop/edge_log/gt_500/gt\415_type.txt 203
C:/Users/Sean/Desktop/edge_log/gt_500/gt\424_type.txt 89
C:/Users/Sean/Desktop/edge_log/gt_500/gt\450_type.txt 1
C:/Users/Sean/Desktop/edge_log/gt_500/gt\458_type.txt 626
C:/Users/Sean/Desktop/edge_log/gt_500/gt\459_type.txt 3786
C:/Users/Sean/Desktop/edge_log/gt_500/gt\470_type.txt 2374
C:/Users/Sean/Desktop/edge_log/gt_500/gt\49_type.txt 5
C:/Users/Sean/Desktop/edge_log/gt_500/gt\54_type.txt 9068
C:/Users/Sean/Desktop/edge_log/gt_500/gt\57_type.txt 5431
C:/Users/Sean/Desktop/edge_log/gt_500/gt\66_type.txt 7987
C:/Users/Sean/Desktop/edge_log/gt_500/gt\7_type.txt 553
"""

"""GT 1000
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\118_type.txt 1809
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\11_type.txt 5
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\181_type.txt 437
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\196_type.txt 2217
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\215_type.txt 2006
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\216_type.txt 1684
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\223_type.txt 211
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\229_type.txt 747
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\245_type.txt 187
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\24_type.txt 319
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\276_type.txt 324
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\311_type.txt 36
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\319_type.txt 2349
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\330_type.txt 195
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\335_type.txt 40
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\342_type.txt 1947
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\350_type.txt 775
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\357_type.txt 2173
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\360_type.txt 56
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\394_type.txt 1704
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\395_type.txt 204
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\398_type.txt 122
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\421_type.txt 1260
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\450_type.txt 3974
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\500_type.txt 225
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\508_type.txt 448
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\511_type.txt 2726
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\518_type.txt 1734
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\538_type.txt 29
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\592_type.txt 2256
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\607_type.txt 918
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\658_type.txt 126
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\690_type.txt 646
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\693_type.txt 195
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\711_type.txt 144
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\717_type.txt 5421
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\718_type.txt 1042
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\723_type.txt 761
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\741_type.txt 1425
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\744_type.txt 907
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\765_type.txt 193
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\773_type.txt 194
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\779_type.txt 820
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\787_type.txt 4072
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\7_type.txt 317
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\816_type.txt 2040
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\817_type.txt 224
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\822_type.txt 89
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\823_type.txt 159
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\834_type.txt 1319
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\837_type.txt 6240
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\852_type.txt 82
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\861_type.txt 118
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\863_type.txt 2560
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\868_type.txt 2856
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\903_type.txt 3786
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\906_type.txt 84
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\960_type.txt 522
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\964_type.txt 304
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\972_type.txt 1550
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\974_type.txt 4299
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\979_type.txt 60
C:/Users/Sean/Desktop/edge_log/gt_1000/gt\997_type.txt 1665
"""