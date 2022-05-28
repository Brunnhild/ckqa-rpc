def v2cPrint(caption,video):
    print("result from v2c")
    print("caption:",caption)
    print("video:",video)
    cms = {}
    cms["caption"] = "caption is :" + caption
    cms["intention"] = "to eat it"
    cms["effect"] = "gets a spoon"
    cms["attribute"] = "hungry"
    cms["need"] = "to gather the ingredients"
    cms["react"] = "happy to have soup"
    queries = [
             'PersonX makes dessert ',
             'PersonX mixes the ingredients together ',
             'PersonX mixes it up '
        ]
    # video_str = "video is :" + str(video)
    # queries = [video_str]
    return cms,queries

if __name__ == '__main__':
    print(v2cPrint("222",1))