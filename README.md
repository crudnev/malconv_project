Cyril Rudnev
April 2026
Instructions on how to re-enact this project:
All you need is VMWare to run the virtual machine! Workstation pro can be downloaded for free.

Link to VM: https://drive.google.com/drive/folders/1Jniam-wSL1yo-hI1tZ0WeAa48MH_L89Y?usp=sharing 

1. Launch VMware workstation. Go to file, open and find the .ova file.
2. Enter a name for the VM and choose the storage location.
3. Click import. 
4. Click power on. Select "I moved it" if prompted. 
5. username: cyril
6. pass: 3ncrypt3d1!
7. I recommend ssh'ing in, its more convenient being able to scroll. Ssh is enabled. Find the VM's IP address by running "ifconfig". Then run "ssh cyril@<ip-address>" on your host machine.
8. Change directories to malconv-evasion-project. This is the project directory. It contains sub directories with
the datasets, scripts, and results. 
9. ~/malconv-evasion-project/results/capa_capabilities contains the output of capa on the original and transformed 
malware samples. 
10. ~/malconv-evasion-project/results/malconv_scores contains all the scores of MalConv ran on original and transformed
malware samples.
11. ~/malconv-evasion-project/datasets/benign - Benign dataset
12. ~/malconv-evasion-project/datasets/malware - Malicious dataset. Has sub directories with original and transformed samples.
13. ~/malconv-evasion-project/models/scripts - Has scripts to score malware and to perform transformations on malware.
14. To run the scripts navigate to the scripts directory 
15. Run this command: source ~/malconv-evasion-project/env/venv/bin/activate
16. To run a script do: python3 technique1v2.py
17. technique1v2.py - Appends 1MB of benign bytes from notepad++.exe to 10 chosen malware samples.
18. technique2.py - Extracts the overlay of anyburn_setup_x64.exe and appends it to 10 chosen malware samples.
19. technique3.py - Modifies fields in the PE header of 10 chosen malware samples.
20. technique4.py - Renames sections inside malware to ones commonly seen in legitimate software.
21. To see the capabilities of malware using capa, navigate to a directory with the samples you want to test.
22. Run "capa <file-name>"
