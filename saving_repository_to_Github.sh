#!/bin/ksh

current_dir=$( cd -P -- "$(dirname -- "$(command -v -- "$0")")" && pwd -P )
print $current_dir
print "yoo Sem what up - different"
git add $current_dir #All files in directory staged for commit
git commit -m "Default"
git push -u
git status

mkdir funnydir
exit

# git remote -v
# Print log of commits
# git log --pretty=format:"%h - %an, %ar : %s"
# git log --pretty=format:"%h %s" --graph
# git remote show origin