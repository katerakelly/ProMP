import click
import doodad
from doodad.launch.launch_api import run_python, run_command
from doodad import mode, mount

### Run ProMP with Docker and doodad on GCP

@click.command()
@click.argument('config', default=None)
@click.option('--zone', default='us-west2-a')
def main(config, zone):
    if config is None:
        print('Must specify config')
        raise(Exception)
    # don't copy these files to remote host because permissions
    ignore_exts = ('.pyc', '.log', '.git', '.mp4', '.viminfo', '.bash_history', '.python_history')
    # in this case, we must manually specify the code mount, because doodad cannot infer it automatically from a general bash command
    code_mnt = mount.MountLocal(local_dir='/home/rakelly/ProMP', mount_point='/root/code', output=False, filter_ext=ignore_exts, pythonpath=True)
    #mujoco_mnt = mount.MountLocal(local_dir='/home/rakelly/ProMP/docker', mount_point='/root/.mujoco', output=False)
    # output will be stored in Google Cloud Storage under gcp_bucket/gcp_log_path/logs/delete_these (if you change this, remember to change root_dir argument in your script call!
    #mujoco_mnt = mount.MountGCP(gcp_path='mujoco', mount_point='/root/.mujoco')
    output_mnt = mount.MountGCP(gcp_path='data', mount_point='/root/code/data', output=True)
    remote = mode.GCPMode(gcp_project='mslac-anusha-kate-tony', \
            gcp_bucket='mslac', \
            gcp_log_path='rl2_{}'.format(config), \
            gcp_image='railrl-nvdocker', \
            gcp_image_project='mslac-anusha-kate-tony', \
            terminate_on_end=True, \
            preemptible=False, \
            zone='{}'.format(zone), \
            instance_type='n1-highmem-16', \
            )

    # the command we want to run: leave the paths alone, I do not know exactly why they have to be this way but that's a problem for later :)
    command = 'source activate meta_mb && export PYTHONPATH=/root/code/_home_rakelly_ProMP:$PYTHONPATH && python /root/code/_home_rakelly_ProMP/run_scripts/rl2_run.py /root/code/_home_rakelly_ProMP/configs/{}.json'.format(config)

    # launch the job! the docker-image will be automatically downloaded from the Docker Hub
    run_command(command, mode=remote, mounts=[code_mnt, output_mnt], verbose=True, docker_image='iclavera/meta-mb:latest')

if __name__=="__main__":
    main()

