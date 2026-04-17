import os

def list_files(directory):
    """
    Menampilkan semua file dalam direktori dan sub-direktori
    
    Args:
        directory (str): Path direktori yang akan diperiksa
    """
    print(f"\n📁 Memeriksa direktori: {directory}")
    print("=" * 60)
    
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        # root: path direktori saat ini
        # dirs: daftar sub-direktori
        # files: daftar file dalam direktori saat ini
        
        if files:
            print(f"\n📂 {root}")
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"   📄 {file} ({file_size:,} bytes)")
                file_count += 1
    
    print(f"\n✅ Total file ditemukan: {file_count}")

# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path direktori yang ingin diperiksa
    target_directory = "."  # "." untuk direktori saat ini
    
    # Pastikan direktori ada
    if os.path.exists(target_directory):
        list_files(target_directory)
    else:
        print(f"❌ Direktori '{target_directory}' tidak ditemukan!")