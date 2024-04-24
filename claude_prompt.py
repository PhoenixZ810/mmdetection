import requests
import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume', '-r', action='store_true', help='resume from the last prompt')
args = parser.parse_args()

def remove_markdown(text):
    # 用正则表达式匹配**加粗**的部分，并替换为去掉**的内容
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # 用正则表达式匹配"引用"的部分，并替换为去掉"的内容
    text = re.sub(r"\"(.*?)\"", r"\1", text)
    # 用正则表达式匹配\n换行符，并替换为空格
    text = re.sub(r"\n", " ", text)
    # 返回处理后的字符串
    return text
def prepare_des(text):
    raw_sents = text.split('\n\n')
    raw_sents.pop()
    if len(raw_sents) == 0:
        raw_sents = text.split('.')
        if raw_sents[-1] == '':
            raw_sents.pop()
            raw_sents.pop()
        else:
            raw_sents.pop()
        description = '.'.join(raw_sents) + '.'
    else:
        description = ' '.join(raw_sents)
    description = remove_markdown(description)
    return description
raw_des0 = "In the center of the image, a vibrant blue lunch tray holds four containers, each brimming with a variety of food items. The containers, two in pink and two in yellow, are arranged in a 2x2 grid.\n\nIn the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface.\n\nAdjacent to it in the top right corner, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple. The colors of the apple slices and pineapple chunks contrast beautifully against the pink container.\n\nBelow these, in the bottom left corner of the tray, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets.\n\nFinally, in the bottom right yellow container, there's a sweet treat - a chocolate chip cookie. The golden-brown cookie is dotted with chocolate chips, their dark color standing out against the cookie's lighter surface.\n\nThe arrangement of these containers on the blue tray creates a visually appealing and balanced meal, with each component neatly separated yet part of a cohesive whole."
raw_des1 = "This image captures a serene moment in a zoo enclosure, where two majestic giraffes are seen in their natural behavior. The giraffes, adorned in their distinctive brown and white patterns, stand tall against the backdrop of lush green trees.\n\nOn the left, one giraffe is actively engaged in a meal, its long neck extended towards the tree as it munches on the verdant leaves. Its companion on the right stands leisurely next to a tree trunk, perhaps taking a break from its own leafy feast.\n\nThe enclosure they inhabit is grassy and spacious, providing them with ample room to roam and forage. The trees dotting the enclosure not only offer a source of food but also create a naturalistic habitat for these towering creatures.\n\nIn summary, this image is a snapshot of life in a zoo, showcasing the grace and beauty of giraffes in an environment designed to mimic their wild habitats."
raw_des2 = "This is a detailed description of the image:\n\n- The image captures a **single zebra** in its natural habitat, a lush green field.\n- The zebra, known for its distinctive **black and white stripes**, is the main subject of the photograph.\n- The zebra is positioned towards the **right side** of the image, giving a side profile view to the observer.\n- It is standing on all **four legs**, indicating a state of calm and ease.\n- The zebra is engaged in a common activity for its species - grazing. Its neck is bent downwards as it feeds on the **green grass** beneath it.\n- The field in which the zebra is grazing is not just green but also dotted with **small white flowers**, adding a touch of contrast and beauty to the scene.\n- There are no other zebras or animals visible in the image, suggesting that this particular zebra might be alone at this moment.\n\nThis description is based on what can be confidently determined from the image. It provides a comprehensive understanding of the image content, including object types, colors, actions, and locations."
raw_des3 = "In the image, there is a woman standing in front of a white sign. The sign has blue text that reads \"WELCOME TO THE LAKE\". The woman is holding a pink umbrella in her hand and is wearing a colorful swimsuit. She is smiling, adding a cheerful vibe to the scene. The background of the image features a serene lake surrounded by lush green trees under a clear blue sky. The woman appears to be the only person in the image, and her position in front of the sign suggests she might be at the entrance of the lake area. The pink umbrella she's holding contrasts nicely with her colorful swimsuit and the natural colors of the background."
raw_des4 = "This image captures a serene moment of a light brown dog, peacefully curled up and sleeping on a white shoe rack. The shoe rack, which is positioned against a brick wall, has three shelves housing at least six pairs of shoes. The shoes vary in style, including casual flip flops, sporty sneakers, and formal dress shoes, and they display a range of colors such as black, white, red, and gray. To the right of the shoe rack, a blue broom leans against the brick wall, standing by for its next use. The scene paints a picture of everyday life with a touch of unexpected sweetness from our canine friend."
raw_des5 = "The image showcases a captivating scene of a dressage routine being performed by two horses and their riders in a grassy field. The horse on the left is a majestic white stallion, while the one on the right is a striking black stallion. Both horses are displaying their strength and agility by rearing up on their hind legs, creating an impressive spectacle.\n\nThe riders, dressed in crisp white outfits and blue hats, appear to be in perfect sync with their horses. Their attire contrasts beautifully with the vibrant green of the field, adding to the overall aesthetic of the image.\n\nIn the background, colorful flags and obstacles can be seen, indicating that this might be a competitive event. The lush trees and shrubs further enhance the natural beauty of the setting.\n\nOverall, this image captures a moment of harmony between the riders and their horses, set against a backdrop of nature's splendor. It's a testament to the skill and grace involved in dressage."
des0 = prepare_des(raw_des0)-
des1 = prepare_des(raw_des1)
des2 = prepare_des(raw_des2)
des3 = prepare_des(raw_des3)
des4 = prepare_des(raw_des4)
des5 = prepare_des(raw_des5)
print(des5)

def generate_payload(prompt, resume):
    if resume:
        payload = json.dumps({"model": "claude-2", "max_tokens": 500, "temperature": 0.01,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"{prompt[0]}"
                                },
                                {
                                    "role": "bot",
                                    "content": " 根据图片描述,我提取出以下名词短语:\n\n[a vibrant blue lunch tray]\n[four containers]\n[two in pink and two in yellow]\n[a slice of bread]\n[a handful of almonds]\n[a rectangle] \n[sliced apples]\n[their fresh white interiors]\n[juicy chunks of pineapple] \n[apple slices]\n[pineapple chunks]\n[a yellow container]\n[a single meatball]\n[broccoli]\n[broccoli florets]\n[a chocolate chip cookie]\n[golden-brown cookie]\n[chocolate chips]\n\n其中:\n['a slice of bread', 'a rectangle'] 指向同一个物体\n['sliced apples', 'apple slices'] 指向同一个物体 \n['juicy chunks of pineapple', 'pineapple chunks'] 指向同一个物体\n['broccoli', 'broccoli florets'] 指向同一个物体\n['a chocolate chip cookie', 'golden-brown cookie'] 指向同一个物体\n\n所以我提取并拼接后的名词短语列表是:\n\n[a vibrant blue lunch tray]\n[four containers] \n[two in pink and two in yellow]\n[a slice of bread, a rectangle]  \n[a handful of almonds]\n[sliced apples, apple slices]\n[their fresh white interiors] \n[juicy chunks of pineapple, pineapple chunks]\n[a yellow container]\n[a single meatball]\n[broccoli, broccoli florets]\n[a chocolate chip cookie, golden-brown cookie] \n[chocolate chips]"
                                },
                                {
                                    "role": "user",
                                    "content": f"{prompt[1]}"
                                },
                                {
                                    "role": "bot",
                                    "content":" 您的反馈非常有价值。我重新整理和改进了一下:\n\n[a vibrant blue lunch tray]\n[four containers]  \n[the top left pink container]\n[a slice of bread] \n[a handful of almonds]\n[sliced apples with their fresh white interiors]\n[juicy chunks of pineapple, pineapple chunks]\n[a yellow container]  \n[a single meatball]\n[broccoli, broccoli florets] \n[a chocolate chip cookie, golden-brown cookie]\n[chocolate chips]\n\n我明白之前的一些提取是不恰当的。这次我重新检查了描述,确保只提取出真正指向具体物体的名词短语,同时也将指向同一物体的多个短语进行了合并。让我知道如果还有需要改进的地方。"
                                },
                                {
                                    "role": "user",
                                    "content": f"{prompt[2]}"
                                },
                                {
                                    "role": "bot",
                                    "content":" 根据图片描述,我提取出以下名词短语:\n\n[two majestic giraffes]\n[their distinctive brown and white patterns]  \n[lush green trees]\n[one giraffe]\n[its long neck]\n[the verdant leaves]\n[Its companion]  \n[a tree trunk]\n[The enclosure they inhabit] \n[grassy and spacious]\n[The trees dotting the enclosure]\n[a source of food]\n[a naturalistic habitat]\n[these towering creatures]\n\n其中:\n['two majestic giraffes', 'one giraffe', 'Its companion'] 指向同一类物体\n['the verdant leaves', 'a source of food'] 指向同一物体\n\n所以我提取并拼接后的名词短语列表是:\n\n[two majestic giraffes, one giraffe, Its companion]  \n[their distinctive brown and white patterns]\n[lush green trees]  \n[its long neck]\n[the verdant leaves, a source of food]\n[a tree trunk]\n[The enclosure they inhabit]  \n[grassy and spacious] \n[The trees dotting the enclosure]\n[a naturalistic habitat]  \n[these towering creatures]"
                                },
                                {
                                    "role": "user",
                                    "content": f"{prompt[3]}"
                                },
                                # {
                                #     "role": "bot",
                                #     "content":" 您的反馈非常有价值。我再次检查并纠正了错误,现在我的理解更加准确了:\n\n[two majestic giraffes] \n[their distinctive brown and white patterns]\n[lush green trees, The trees dotting the enclosure]\n[one giraffe on the left]\n[its long neck]  \n[the verdant leaves]\n[Its companion on the right] \n[a tree trunk] \n[The enclosure they inhabit]\n\n我明白之前对一些短语的处理是不当的:\n\n1. 'grassy and spacious' 不是名词短语,已删除\n2. 'a naturalistic habitat' 是抽象概念,不是具体物体,已删除\n3. 'lush green trees'和'The trees dotting the enclosure'指向同一类物体,已合并\n4. 忽略了'a source of food' \n5. 将'one giraffe'和'Its companion'分别细化为'one giraffe on the left'和'Its companion on the right',以示区分\n\n我会注意这些问题,改进提取名词短语的能力。非常感谢您的耐心指导!"
                                # },
                                # {
                                #     "role": "user",
                                #     "content": f"{prompt[4]}"
                                # },

                            ]})
    if not resume:
        payload = json.dumps({"prompt": prompt, "model": "claude-2", "max_tokens": 500, "temperature": 0})
    return payload

url = "https://openxlab.org.cn/gw/alles-apin-hub/v1/claude/v1/text/chat"

# prompt = f"你现在要从图片描述中提取出具体的、指向实际物体的名词或者短语短句，如: 'four containers, 'a vibrant blue lunch tray', 'fruit','a single meatball'等， 并排除其中抽象名词短语以及不指向真实物体的、不能用框来指定的名词短语，比如'happy', 'love', 'world', 'the space', 'time', 'the top right corner', 'center of the image'等。此外，'broccoli florets'和'broccoli'其实是同一个物体，'a chocolate chip cookie'和'golden-brown cookie'也是同一个物体，对于有不同描述的同一个物体，比如'broccoli florets'和'broccoli'其实是同一个物体，'a chocolate chip cookie'和'golden-brown cookie'也是同一个物体，你应该将这个物体的不同描述拼接成一个列表，比如['broccoli florets', 'broccoli']，['a chocolate chip cookie', 'golden-brown cookie']。现在图片描述如下: {des} 按照我的要求提取其中的名词短语。"
# prompt0 = f"你是一个语言学专家，现在给定图片描述，你要先理解整段描述，然后从中提取出具体的、指向实际物体的名词或者短语短句，如: 'four containers, 'a vibrant blue lunch tray', 'fruit','a single meatball', 'a slice of bread', 'a handful of almonds'等。如果同一个物体有不同的名词短语，比如'broccoli florets'和'broccoli'其实是同一个物体，'a chocolate chip cookie'和'golden-brown cookie'也是同一个物体，你应该理解图片描述，然后将指向同一个物体的不同描述拼接成一个列表，比如['broccoli florets', 'broccoli']，['a chocolate chip cookie', 'golden-brown cookie']直接输出，不要单独输出。同时你还应该排除句子中的抽象名词短语，比如'happy', 'love',并且排除不可见的、不能用框来指定的名词短语，比如 'world', 'the space', 'wind', 'time', 'the top right corner', 'center of the image'等。现在图片描述如下: {des0} 按照我的要求提取其中的名词短语。"

prompt0 = """
You are a linguist. Now you are given a picture description, you need to first understand the whole description, and then extract specific nouns or phrases that point to actual objects from it. We use json format for output, the keys include 'Input' and 'Output', here is a template:
```
{
    'Input': [**picture description**],
    'Output': [**nouns or phrases list you extract from picture description**]
}
```
The rules are as follows:
1. If the same object has different noun phrases, such as 'broccoli florets' and 'broccoli' are actually the same object, 'a chocolate chip cookie' and 'a golden-brown cookie' are also the same object, you should understand the picture description, if one of the phrases is much simpler than the other, you should give up the simple one, such as 'the field' and 'a lush green field', you should give up 'the field', else then concatenate the different descriptions pointing to the same object into a list, such as ['broccoli florets', 'broccoli'], ['a chocolate chip cookie', 'a golden-brown cookie'] and output them directly, do not output them separately.
2. You should exclude abstract noun phrases in the sentence, such as 'happy', 'love'
3. You should exclude noun phrases which are invisible and cannot be specified by a box, such as 'world', 'the habitat', 'the behavior', 'the space', 'wind', 'time', 'the top right corner', 'center of the image', 'right side of the image', etc.

Here are three generation examples:
```
{
    'Input': [%s],
    'Output': [['a vibrant blue lunch tray'], ['four containers'], ['the top left pink container'], ['bread'], ['almonds'], ['sliced apples with their fresh white interiors'], ['juicy chunks of pineapple', 'pineapple chunks'], ['a yellow container'], ['a single meatball'], ['broccoli, broccoli florets'], ['a chocolate chip cookie', 'golden-brown cookie'], ['chocolate chips']]
}
```
```
{
    'Input': [%s],
    'Output': [['two majestic giraffes'], ['distinctive brown and white patterns'], ['one giraffe on the left'], ['Its companion on the right'], ['lush green trees'], ['its long neck'], ['the verdant leaves'], ['a tree trunk'], ['The enclosure they inhabit']]
}
```
```
{
    'Input': [%s],
    'Output': [['a single zebra', 'The zebra'], ['a lush green field'], ['its distinctive black and white stripes'], ['all four legs'], ['neck'], ['the green grass beneath it'], ['small white flowers']]
}
```
Please start engineering. Given {'Input': [%s]}. Output:
""" % (des0, des1, des2, des5)


# prompt1 = "你做的很不错，但是还有地方需要改进。首先，你遗漏了'the top left pink container'；第二，'two in pink and two in yellow'不指向具体的一个物体，应该去除；第三，'Sliced apples with their fresh white interiors'不应该被拆分为'sliced apples'和'their fresh white interiors'；第四，'a rectangle'是'The bread is cut into a rectangle'的一部分，不能单独提取出来。你明白了吗？"
# # prompt1 = "你的结果遗漏了一些物体，比如'a slice of bread', 'a handful of almonds'"
# # prompt = "你的结果遗漏了一些物体，比如'a slice of bread', 'a handful of almonds'; 此外，'broccoli florets'和'broccoli'其实是同一个物体，'a chocolate chip cookie'和'golden-brown cookie'也是同一个物体，对于有不同描述的同一个物体，你应该将这个物体的不同描述拼接成一个列表，比如['broccoli florets', 'broccoli']，['a chocolate chip cookie', 'golden-brown cookie']。请重新抽取所有具体的，指向实际物体的名词短语。]"
# prompt2 = f"按照上述的方法，提取以下图片描述：{des1}"
# # prompt = f"重新抽取以下这段描述：{des}"

# prompt3 = f"你的回答有误，首先，你提取出的应该是名词，'grassy and spacious'是形容词；第二，你应该删除所有不是具体物体对象的名词短语，'a naturalistic habitat'中的'habitat'不是一个具体的物体对象，应该删除；第三，'lush green trees'和'The trees dotting the enclosure'描述的是同一个物体，第四，'the verdant leaves'和'a source of food'虽然在句子中语义指向的是同一个物体，但是'a source of food'描述过于简略，因此应该忽略；第五，'one giraffe'和'Its companion'描述的是不同的对象，因此在提取的时候应该做区分，比如'On the left, one giraffe is actively engaged in a meal'中应该提取出'one giraffe on the left'，同理，'Its companion on the right stands leisurely next to a tree trunk'中也应该提取出'Its companion on the right'。你明白了吗？"

# prompt4 = f"按照上述的方法，提取以下图片描述：{des2}"

if args.resume:
    payload = generate_payload([prompt0, prompt1, prompt2, prompt3, prompt4], args.resume)
else:
    payload = generate_payload(prompt0, args.resume)

headers = {
    'alles-apin-token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo5MywidXNlcm5hbWUiOiLotbXnv5TlrociLCJhcHBseV9hdCI6MTcwMTE0NDA4ODYwOSwiZXhwIjoxNzA5MDkyODg2ODcwfQ.5nRnd4kuHSzAgHat8Zc8sKJ3E94Bxq0my4c71SddSX4',
    'Content-Type': 'application/json',
}

response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)
output_info = json.loads(response.text)
output = output_info['data']['completion']
dict = output.split('```')[1]
match = re.search(r"'Input': \[(.*?)\]", dict, re.DOTALL)
if match:
    input_content = match.group(1)
    print(input_content)
match = re.search(r"'Output': \[(.*?)\]\]", dict, re.DOTALL)
if match:
    output_content = match.group(1)
    print(output_content)
# dict = eval(dict)
# dict.replace('\n', '')
# print(dict)
# dict = json.loads(dict)
# print(dict)