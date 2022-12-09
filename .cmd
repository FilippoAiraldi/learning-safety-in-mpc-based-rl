autoflake --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --in-place --recursive .
isort .
black .
