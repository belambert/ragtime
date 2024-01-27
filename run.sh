# poetry run python ./ragtime/train_translate.py
# poetry run python ./ragtime/main.py \"what are asters?\""

gcloud batch jobs submit --job-prefix ragtime --location us-central1 --config - <<EOD
{
  "taskGroups": [
    {
      "taskCount": "1",
      "parallelism": "1",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "8000",
          "memoryMib": "16000",
          "bootDiskMib": "10000"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "us-docker.pkg.dev/llm-exp-405305/ragtime/main:latest",
              "entrypoint": "/bin/sh",
              "commands": [
                "-c",
                "poetry run python ./ragtime/train.py"
              ],
              "volumes": []
            }
          }
        ],
        "volumes": [
           {
             "deviceName": "wiki-dpr",
             "mountPath": "/mnt/disks/data"
           }
         ]
      }
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "installGpuDrivers": true,
        "policy": {
          "accelerators": [
            {
              "type": "nvidia-l4",
              "count": 1
            }
          ],
          "disks": [
            {
              "deviceName": "wiki-dpr",
              "existingDisk": "projects/llm-exp-405305/zones/us-central1-a/disks/wiki-dpr"
            }
          ]
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
