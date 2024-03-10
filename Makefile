build:
	podman build -t pynucleus-binder .

test1:
	podman run -it --rm -p 8889:8888 pynucleus-binder jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888

test2:
	podman run -it --rm pynucleus-binder bash -c 'echo "whoami:" `whoami`; echo "id-u:  " `id -u`; echo "pwd:   " `pwd`; echo "HOME:"; ls -alh ~'
