mc find s3/projet-slums-detection/data-raw/PLEIADES/GUYANE/2024 --older-than "2d" --exec "mc rm --force {}"
mc find s3/projet-slums-detection/data-raw/PLEIADES/GUYANE/2024 --smaller "300k" --exec "mc rm --force {}"
