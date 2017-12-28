#!/bin/sh -e

set -o nounset

echo "DOWNLOAD SCRIPT FOR YT VIDEOS"
echo "Usage: get_videos yt_url_to_playlist directory_to_save_files_to"
echo

# For example https://www.youtube.com/playlist?list=PLalVOiLxaOcXR6YJlps8_WwIIxIYk2IQC
URL=$1
OUT_DIR=$2

mkdir -p $OUT_DIR
cd $OUT_DIR
youtube-dl --continue --ignore-errors -f bestvideo -o '%(id)s.%(ext)s' $URL


# The following renames the files if they were downloaded with '%(title)s.%(id)s.%(ext)s'
# and save the titles in an index file. However, with this setup, you cannot run
# with --continue setting

# touch $OUT_DIR/index.txt
# find . -type f | while read F; do
#     clip_name=$(echo $F | awk -F'.' '{for(i=1;i<NF-1;++i){printf "%s", $i; if (i < NF-2) { printf "."; }}}')
#     new_fname=$(echo $F | awk -F'.' '{printf "%s.%s", $(NF-1), $NF;}')

#     RENAMED=$(grep "$new_fname" $OUT_DIR/index.txt)
#     if [[ ! -z "$RENAMED" ]]; then
#         echo "File $clip_name already renamed"
#     else
#         echo "$clip_name  ---->   $new_fname"
#         mv "$F" "$new_fname"
#         echo "$new_fname $clip_name" >> $OUT_DIR/index.txt
#     fi
# done
