#!/usr/bin/env python3
import os
import stat
import paramiko
from getpass import getpass

'''
    This programm allows to send custom sounds to the robot
'''

# ——————————————————————————————————————————————
# CONFIGURATION
# ——————————————————————————————————————————————
HOSTNAME   = "10.10.10.10"
USERNAME   = "niryo"
PASSWORD   = "robotics"
LOCAL_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Audio'))
REMOTE_DIR = "niryo_robot_saved_files/niryo_robot_user_sounds"

# ——————————————————————————————————————————————
# FONCTIONS
# ——————————————————————————————————————————————
def ensure_remote_dir(sftp, remote_path: str):
    """
        Creating remote_path if needed
    """
    parts = remote_path.split('/')
    curr = ""
    for part in parts:
        curr = curr + "/" + part if curr else part
        try:
            sftp.stat(curr)
        except IOError:
            sftp.mkdir(curr)
            print(f"[mkdir] {curr}")

def upload_all_wavs(sftp, local_dir: str, remote_dir: str):
    """
        Upload every .wav of local_dir to remote_dir towards robot.
    """
    for fname in os.listdir(local_dir):
        if not fname.lower().endswith('.wav'):
            continue
        local_path  = os.path.join(local_dir, fname)
        remote_path = f"{remote_dir}/{fname}"
        print(f"[upload] {fname} → {remote_path}")
        sftp.put(local_path, remote_path)

def main():
    global PASSWORD
    if PASSWORD is None:
        PASSWORD = getpass(f"Password for {USERNAME}@{HOSTNAME}: ")

    # 1. Création du client SSH + connexion
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connection to {USERNAME}@{HOSTNAME}…")
    ssh.connect(HOSTNAME, username=USERNAME, password=PASSWORD)

    # 2. Ouverture du canal SFTP
    sftp = ssh.open_sftp()

    # 3. S’assurer que le dossier existe
    ensure_remote_dir(sftp, REMOTE_DIR)

    # 4. Upload des fichiers
    upload_all_wavs(sftp, LOCAL_DIR, REMOTE_DIR)

    # 5. Fermeture
    sftp.close()
    ssh.close()
    print("Done :  .wav files copied.")

if __name__ == "__main__":
    main()
