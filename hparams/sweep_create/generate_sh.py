import os

# Đường dẫn thư mục chứa các config .json
sweep_dir = "../sweep_STGM_grid_split"
script_output_dir = "../sweep_STGM_scripts"
os.makedirs(script_output_dir, exist_ok=True)

for computer in sorted(os.listdir(sweep_dir)):
    computer_path = os.path.join(sweep_dir, computer)
    if not os.path.isdir(computer_path):
        continue

    # Với computer1, computer2: part folders nằm ngay bên trong
    if computer != "computer3":
        for part in sorted(os.listdir(computer_path)):
            part_path = os.path.join(computer_path, part)
            if not os.path.isdir(part_path):
                continue

            script_lines = [
                "#!/bin/bash",
                f"# Sweep for: {computer}/{part}",
                "",
            ]

            for json_file in sorted(os.listdir(part_path)):
                if json_file.endswith(".json"):
                    json_path = os.path.join("hparams", "sweep_STGM_grid_split", computer, part, json_file)
                    command = f"python system/main.py --cfp ./{json_path} --wandb True"
                    script_lines.append(command)

            script_name = f"{computer}_{part}.sh"
            script_file_path = os.path.join(script_output_dir, script_name)
            with open(script_file_path, "w") as f:
                f.write("\n".join(script_lines))

    # Với computer3: cần đi sâu thêm vào gpu folders
    else:
        for gpu_folder in sorted(os.listdir(computer_path)):
            gpu_path = os.path.join(computer_path, gpu_folder)
            if not os.path.isdir(gpu_path):
                continue

            for job_folder in sorted(os.listdir(gpu_path)):
                job_path = os.path.join(gpu_path, job_folder)
                if not os.path.isdir(job_path):
                    continue

                script_lines = [
                    "#!/bin/bash",
                    f"# Sweep for: {computer}/{gpu_folder}/{job_folder}",
                    "",
                ]

                for json_file in sorted(os.listdir(job_path)):
                    if json_file.endswith(".json"):
                        json_path = os.path.join("hparams", "sweep_STGM_grid_split", computer, gpu_folder, job_folder, json_file)
                        command = f"python system/main.py --cfp ./{json_path} --wandb True"
                        script_lines.append(command)

                script_name = f"{computer}_{gpu_folder}_{job_folder}.sh"
                script_file_path = os.path.join(script_output_dir, script_name)
                with open(script_file_path, "w") as f:
                    f.write("\n".join(script_lines))
