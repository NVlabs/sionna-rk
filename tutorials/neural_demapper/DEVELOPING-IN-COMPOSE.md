# Developing in oai-gnb

## Fixing missing linker error messages in `docker build ran-build`

In `./cmake_targets/tools/build_helper`, change the reprinted error context in the `compilations()` function from
```
egrep -A3 "warning:|error:" $dlog/$logfile || true
```
to
```
egrep -C6 "warning:|error:" $dlog/$logfile || true
```
in order to include context before and after lines including 'error' (`ld` failures generate such lines after the actual error message).

## Running a debugger (`gdb` and `VS code`)

To run a `gdbserver` inside the `gNB` container to attach debuggers to, we override the `docker compose` command line, e.g. as follows:

docker-compose.override.yaml
```
services:
  oai-gnb:
    command: ["gdbserver",":7777","/opt/oai-gnb/bin/nr-softmodem","-O","/opt/oai-gnb/etc/gnb.conf"]
```

Note that we simply prepended `gdbserver :7777` to the pre-existing command line of the `oai-gnb` container, which can be obtained as follows:
```
$ docker inspect --format='{{json .Config.Cmd}}' oai-gnb
["/opt/oai-gnb/bin/nr-softmodem","-O","/opt/oai-gnb/etc/gnb.conf"]
```

We can then attach using `gdb` as follows:
```
$ gdb
> target remote container.ip:7777
```
To get the local IP of a running container, run:
```
$ docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container-name, e.g. oai-gnb>
```

To attach using the `VS code` debugger, the following example `launch.json` configuration can be used:
```
{
    "version": "0.2.0",
    "configurations": [
      {
        "name": "C++ Remote Debug",
        "type": "cppdbg",
        "request": "launch",
        "program": "/opt/oai-gnb/bin/nr-softmodem",
        "customLaunchSetupCommands": [
          { "text": "target remote container.ip:7777", "description": "attach to target", "ignoreFailures": false }
        ],
        "launchCompleteCommand": "None",
        "stopAtEntry": false,
        "cwd": "/",
        "environment": [],
        "externalConsole": false,
        "pipeTransport": { // needed if you are working on a different maching than the container host
            "pipeCwd": "${workspaceRoot}",
            "pipeProgram": "ssh",
            "pipeArgs": [
                "user@remote.ip"
            ],
            "debuggerPath": "/usr/bin/gdb"
        },
        "sourceFileMap": {
                "/oai-ran":"${workspaceRoot}"
        },
        "targetArchitecture": "arm",
        "linux": {
          "MIMode": "gdb",
          "miDebuggerServerAddress": "container.ip:7777",
          "setupCommands": [
            {
              "description": "Enable pretty-printing for gdb",
              "text": "-enable-pretty-printing",
              "ignoreFailures": true
            }
          ]
        }
      }
}
```

## Running TRT demapper

Limiting thread pool to one core (5) for now, limiting encoding to QAM16 and loading the TRT-based demapper library.

.env
```
GNB_EXTRA_OPTIONS=--thread-pool 5 --MACRLCs.[0].ul_max_mcs 14 --loader.demapper.shlibversion _trt
```

### Mounting models and config

docker-compose.override.yaml
```
services:
   oai-gnb:
       volumes:
       - ./models/:/opt/oai-gnb/models
       - ./demapper_trt.config:/opt/oai-gnb/demapper_trt.config
```

Pre-trained models are in `tutorials/neural_demapper/tests/models`.

### TRT config file format:

Schema:
```
<trt_engine_file:string>
<trt_normalized_inputs:int>
```
e.g.
```
models/neural_demapper_qam16_2.plan
1
```

## Inspecting and debugging inside a container

First, find out the commands that are usually run inside the container of interest, e.g.:
```
$ docker inspect --format='{{json .Config.Entrypoint}} {{json .Config.Cmd}}' oai-gnb
["/tini","-v","--","/opt/oai-gnb/bin/entrypoint.sh"] ["/opt/oai-gnb/bin/nr-softmodem","-O","/opt/oai-gnb/etc/gnb.conf"]
```

We can then override the entrypoint to run an interactive session instead of the default launch procedure:

docker-compose.override.yaml
```
    oai-gnb:
         stdin_open: true # docker run -i
         tty: true        # docker run -t
         entrypoint: /bin/bash
```


To attach to a running session after running `./start_system.sh` or `docker compose up -d oai-gnb`, and for example start a debug session, run:
```
$ docker container attach oai-gnb
$ gdb --args /tini -v -- /opt/oai-gnb/bin/entrypoint.sh  ./bin/nr-softmodem -O /opt/oai-gnb/etc/gnb.conf
```

## Debugging: Running memcheck in interactive docker compose

We can use the `compute-sanitizer` tool from the NVIDIA Cuda Toolkit to run a GPU memcheck inside an interactive container session launched as above with the following commands:
```
$ docker container attach oai-gnb
$ /tini -v -- /opt/oai-gnb/bin/entrypoint.sh compute-sanitizer --require-cuda-init=no --tool memcheck ./bin/nr-softmodem -O /opt/oai-gnb/etc/gnb.conf
```
