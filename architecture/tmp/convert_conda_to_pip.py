input_file = "conda_requirements.txt"
output_file = "requirements.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if "=" in line:
            package, version, _ = line.split("=")
            outfile.write(f"{package}=={version}\n")
