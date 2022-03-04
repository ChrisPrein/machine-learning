ECHO uploading package to testpypi

python -m twine upload --verbose --repository testpypi dist/*

ECHO upload done