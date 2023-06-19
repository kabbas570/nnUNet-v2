import os

def main():
    
    folder_path = r'C:\My_Data\SEG.A. 2023\sing\dummy/'
    for count,filename in enumerate(sorted(os.listdir(folder_path))):
        dst ="SEGA_" + str(count).zfill(3) + ".nii.gz"
        
        src = f"{folder_path}/{filename}"
        dst = f"{folder_path}/{dst}"
        
        os.rename(src,dst)
        
      
if __name__ == '__main__':
    
    main()
