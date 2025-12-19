Add the contents on `env.changes` to the file `.env` in the local docker-compose configuration that you are using. The settings will limit the UL MCS to QAM16 and instruct the code to use the libdemapper_orig plugin library.

Create the demapper_in.txt and demapper_out.txt files (for example, using touch) and make sure it has world write access (eg: chmod 777 demapper_in.txt)
Then, add the docker compose override file in order to map the in/out files for capture to the gNB image.