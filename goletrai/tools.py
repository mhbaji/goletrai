
import json
import os

import requests


class JSON:
    @staticmethod
    def save(file_path:str, data:dict):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def read(file_path:str):
        data = {}
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    
class Models():
    @staticmethod
    def download(url:str, modelpath:str):
        try:
            print('Mulai Unduh Model ...')
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 1024

            with open(modelpath, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    percent = downloaded * 100 / total_size
                    print(f"\rMengunduh: {percent:.2f}%", end="")
            print("\nTerunduh.")
        except Exception as e:
            print(f"error: {e}")

    @staticmethod
    def update(cachePath:str, modelpath:str):
        remoteUrl = 'https://raw.githubusercontent.com/mhbaji/goletrai/refs/heads/main/goletrai/src/models.json'
        remoteData = {}
        try:
            print("Unduh Dokumen Models")
            remoteData = requests.get(remoteUrl).json()
            print("Selesai Unduh")
        except Exception as e:
            print(f"Terdapat Error: {e}")
            exit()

        if not os.path.exists(cachePath):
            print("File Cache Tidak Ditemukan")
            Models.download(remoteData['history'][remoteData['version']]['url'], modelpath)
            JSON.save(cachePath, remoteData)
            print("Simpan File Cache")
        else:
            cacheData = JSON.read(cachePath)
            if cacheData['version'] != remoteData['version']:
                print(
                    f"""Versi Model Sekarang: {cacheData['version']},\n
                    Sudah Tersedia Versi Terbaru yaitu {remoteData['version']}"""
                    )
                Models.download(remoteData['history'][remoteData['version']]['url'], modelpath)
                JSON.save(cachePath, remoteData)
            else:
                print(f"Model Sudah Versi Terbaru yaitu {cacheData['version']}")