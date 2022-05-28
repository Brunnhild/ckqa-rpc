import sys
import json
import torch
import os
print(os.getcwd())
sys.path.append('./HybridNet/')
print(os.getcwd())
from opts import *
from one_video_add import data_extract
from test_cms import main
def process(query,video):
    #往数据集中添加数据
    #new_caption可以从网页前端获得
    #new_caption = "kids watching sports on tv"#"a person is using a spoon to mix a dessert in a bowl"
    new_caption = query
    queries=data_extract(new_caption)
    
    opt = parse_opt()
    opt = vars(opt)
    opt['cuda'] = True
    opt['captions'] = json.load(open(opt['caption_json']))
    opt['batch_size'] = 30
    #将待检测视频传入函数
    opt['video_to_test'] = video
    #intention
    opt['cms'] = 'int'
    int_cms = main(opt)
    #effect
    opt['cms'] = 'eff'
    eff_cms = main(opt)
    #attribute
    opt['cms'] = 'att'
    att_cms = main(opt)
    #need
    opt['cms'] = 'need'
    need_cms = main(opt)
    #react
    opt['cms'] = 'react'
    react_cms = main(opt)

    cms = {"intention":int_cms,"effect":eff_cms,"attribute":att_cms,"need":need_cms,"react":react_cms}

    print("cms:",cms)

    return cms,queries

if __name__ == '__main__':
    #opt = parse_opt()
    #opt = vars(opt)
    #opt['cuda'] = True
    #opt['captions'] = json.load(open(opt['caption_json']))
    #opt['batch_size'] = 30
    video = 1 #opt['video_to_test']
    #print("\nvideo to test:",opt['video_to_test'])
    #print('\n')
    #process(video,opt)
    query = "kids watching sports on tv"
    process(query,video)
