while read line; do
  # 用制表符分割每一行，获取文件名和链接
  file_name=$(echo $line | cut -f1)
  cdn_link=$(echo $line | cut -f2)
  # 下载链接指向的文件，并保存为文件名
  wget -c -O $file_name $cdn_link
done < '/mnt/workspace/zhaoxiangyu/code_new/grounding_mm_mine/sam_list.log'
