import os
import random
import json
import Image
"""
1. go through each directory.
2. save the json
{
    classes: {
        'class name': 1,
        'class name 2': 2,
        ...
    },
    videos: {
        'name_of_video': {
                class: 1,
                classname: 'class name',
            }
        ...
    },
    frames: [
        {
            path: "path/to/the/file",
            thumb: "path/to/the/thumb",
            video: "name_of_video",
            class: 1,
            classname: 'class name',
        },
        ...
    ],
}
"""

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data/images')
thumb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data/thumbs')
train_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.txt')
val_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'val.txt')
def create_thumbs(todo):
    for frame in todo['frames']:
        if os.path.exists(frame['thumb']):
            continue
        try:
            head, tail = os.path.split(frame['thumb'])
            if not os.path.exists(head):
                os.makedirs(head)
            im = Image.open(frame['path'])
            im.resize((256,256)).save(frame['thumb'])
        except Exception as e:
            raise e


def get_todo():
    """generate the classes"""
    classes={}
    for idx, name in enumerate(os.listdir(data_dir)):
        classes[name]=idx

    """get the frames and videos"""
    frames=[]
    videos={}
    for dirName, subdirList, fileList in os.walk(data_dir):
        for jpg in fileList:
            if '.jpg' == os.path.splitext(jpg)[-1]:
                classname = os.path.split(dirName)[-1]
                video = '_'.join(jpg.split('_')[:-1])
                if video not in videos:
                    videos[video]={
                            'class': classes[classname],
                            'classname': classname,
                        }
                frames.append({
                    'path': os.path.abspath(os.path.join(dirName, jpg)),
                    'thumb': os.path.abspath(os.path.join(dirName.replace(data_dir,thumb_dir), jpg)),
                    'video': video,
                    'class': classes[classname],
                    'classname': classname,
                    })
    todo = {
        'classes':classes,
        'frames':frames,
        'videos':videos,
        }
    return todo

def make_txts(todo):
    """do a 70/30 split"""
    videos_by_class = {x:[] for x in todo['classes']}
    for video, prop in todo['videos'].iteritems():
       videos_by_class[prop['classname']].append(video)
    for classname, videolist in videos_by_class.iteritems():
        '''fixed seed here'''
        random.seed(42)
        random.shuffle(videolist)
        split = int(len(videolist)*.7)
        train = videolist[:split]
        val = videolist[split:]
        """mark the videos as train/val"""
        for video in train:
            todo['videos'][video]['type']='train'
        for video in val:
            todo['videos'][video]['type']='val'
    """make the text files"""
    txt_files={
        'train':open(train_txt, 'w'),
        'val':open(val_txt, 'w'),
        }
    for frame in todo['frames']:
        txt_files[todo['videos'][frame['video']]['type']].write("{} {}\n"
            .format(frame['thumb'], frame['class']))
    for txt in txt_files:
        txt_files[txt].close()




if __name__ == '__main__':
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'todo.json')):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'todo.json'), 'r') as f:
            todo = json.loads(f.read())
    else:
        todo=get_todo()
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'todo.json'), 'w') as f:
            f.write(json.dumps(todo))
    """do a 70/30 split of train and val sets"""
    make_txts(todo)
    """create thumbs"""
    create_thumbs(todo)

