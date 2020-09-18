import argparse
import os
import subprocess
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


if __name__ == "__main__":

    import evalresults

    print("Testing MOOD docker image...")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--docker_name", required=False, type=str, help="Name of the docker image you want to test",
        default='mood_submission_5'
    )
    # parser.add_argument(
    #     "-d", "--docker_name", required=False, type=str, help="Name of the docker image you want to test",
    #     default='test'
    # )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=False,
        type=str,
        help=(
            "Data dir, it will require to contain a folder 'brain' and 'abdom' which will both "
            "each require a subfolder 'toy' and 'toy_label' i.e. data_dir/brain/toy,"
            " data_dir/brain/toy_label, data_dir/abdom/toy, data_dir/abdom/toy_label"
        ),
        default='/home/zhai/文档/docker_test'
    )
    parser.add_argument(
        "-o", "--output_dir", required=False, type=str, help="Folder where the output/ predictions will be written too",
        default='/home/zhai/文档/docker_test/output'
    )
    parser.add_argument(
        "-t", "--task", required=False, choices=["sample", "pixel"], type=str, help="Task, either 'pixel' or 'sample' ",
        default='pixel'
    )
    parser.add_argument(
        "--no_gpu",
        required=False,
        default=False,
        type=bool,
        help="If you have not installed the nvidia docker toolkit, set this arg to False",
    )

    args = parser.parse_args()

    docker_name = args.docker_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    task = args.task
    no_gpu = args.no_gpu

    tmp_dir = None
    if output_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        output_dir = tmp_dir.name

    brain_data_dir = os.path.join(input_dir, "brain")
    abdom_data_dir = os.path.join(input_dir, "abdom")

    if not os.path.exists(brain_data_dir):
        print(f"Make sure there is a 'brain' folder in your input_dir, i.e. {brain_data_dir}")
        exit(1)
    if not os.path.exists(abdom_data_dir):
        print(f"Make sure there is a 'abdom' folder in your input_dir, i.e. {abdom_data_dir}")
        exit(1)

    brain_toy_data_dir = os.path.join(brain_data_dir, "data")
    abdom_toy_data_dir = os.path.join(abdom_data_dir, "data")

    brain_toy_label_dir = os.path.join(brain_data_dir, "label", task)
    abdom_toy_label_dir = os.path.join(abdom_data_dir, "label", task)

    if not os.path.exists(brain_toy_data_dir) or not os.path.exists(brain_toy_label_dir):
        print(f"Make sure there is a 'toy' and 'toy_label' folder in your brain_dir ({brain_data_dir})")
        exit(1)
    if not os.path.exists(abdom_toy_data_dir) or not os.path.exists(abdom_toy_label_dir):
        print(f"Make sure there is a 'toy' and 'toy_label' folder in your abdom_dir ({abdom_data_dir})")
        exit(1)

    output_brain_dir = os.path.join(output_dir, "brain")
    output_abdom_dir = os.path.join(output_dir, "abdom")

    os.makedirs(output_brain_dir, exist_ok=True)
    os.makedirs(output_abdom_dir, exist_ok=True)

    gpu_str = ""
    if not no_gpu:
        gpu_str = "--gpus all "

    print("\nPredicting brain data...")

    ret = ""
    try:
        docker_str = (
            f"sudo docker run {gpu_str}-v {brain_toy_data_dir}:/mnt/data "
            f"-v {output_brain_dir}:/mnt/pred --read-only {docker_name} sh /workspace/run_{task}_brain.sh /mnt/data /mnt/pred"
        )

        # docker_str = (
        #     f"sudo docker run {gpu_str}-v {brain_toy_data_dir}:/mnt/data "
        #     f"-v {output_brain_dir}:/mnt/pred --read-only {docker_name} sh /workspace/debug.sh /mnt/data /mnt/pred"
        # )

        ret = subprocess.run(docker_str.split(" "), check=True,)
    except Exception:
        print(f"Running Docker brain-{task}-script failed:")
        print(ret)
        exit(1)

    print("Predicting abdominal data...")

    try:
        docker_str = (
            f"sudo docker run {gpu_str}-v {abdom_toy_data_dir}:/mnt/data "
            f"-v {output_abdom_dir}:/mnt/pred --read-only {docker_name} sh /workspace/run_{task}_abdom.sh /mnt/data /mnt/pred"
        )
        ret = subprocess.run(docker_str.split(" "), check=True,)
    except Exception:
        print(f"Running Docker abdom-{task}-script failed:")
        print(ret)
        exit(1)

    print("\nEvaluating predictions...")

    brain_score = evalresults.eval_dir(
        output_brain_dir, brain_toy_label_dir, mode='pixel', save_file=os.path.join(output_dir, "brain_pixel.txt")
    )
    brain_score_2 = evalresults.eval_dir(
        output_brain_dir, brain_toy_label_dir, mode='sample', save_file=os.path.join(output_dir, "brain_sample.txt")
    )
    print("Brain-pixel score:", brain_score)
    print("Brain-sample score:", brain_score_2)

    abdom_score = evalresults.eval_dir(
        output_abdom_dir, abdom_toy_label_dir, mode='pixel', save_file=os.path.join(output_dir, "abdom_pixel.txt")
    )
    abdom_score_2 = evalresults.eval_dir(
        output_abdom_dir, abdom_toy_label_dir, mode='sample', save_file=os.path.join(output_dir, "abdom_sample.txt")
    )
    print("Abdominal-pixel score:", abdom_score)
    print("Abdominal-sample score:", abdom_score_2)

    if tmp_dir is not None:
        tmp_dir.cleanup()

    print("Done.")
