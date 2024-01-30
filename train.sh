# script to launch training on a GCP Batch machine with a GPU and attached disk holding
# the wiki_dpr data

gcloud batch jobs submit --job-prefix ragtime-train --location us-central1 --config - <<EOD
{
  "taskGroups": [
    {
      "taskCount": "1",
      "parallelism": "1",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "8000",
          "memoryMib": "64000",
          "bootDiskMib": "10000"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "us-docker.pkg.dev/llm-exp-405305/ragtime/main:latest",
              "entrypoint": "/bin/sh",
              "commands": [
                "-c",
                "poetry run train --wandb --batch-size 8"
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
