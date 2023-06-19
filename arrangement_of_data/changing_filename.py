import os
def main():
    
    folder_path = r'C:\My_Data\SEG.A. 2023\nn_format'
    for count, filename in enumerate(sorted(os.listdir(folder_path))): # filename= folder_name
        
        dst ='SEGA_'+ str(count).zfill(3) 
                
        src = f"{folder_path}/{filename}"
        
        
        files_src = (os.listdir(src))
        
        dst = f"{folder_path}/{dst}"
        
        os.rename(src,dst)
        
        src_abs_img = dst + '/' +files_src[0] 
        src_abs_gt = dst + '/' +files_src[1] 
        
        
        dst_abs_img  = dst +'/' + 'SEGA_'+ str(count).zfill(3) +'_' + files_src[0] 
        
        
        dst_abs_gt  = dst +'/' + 'SEGA_'+ str(count).zfill(3) +'_' + files_src[1][:-9] + '_gt.nrrd'
        
        
        # #dst_abs_img  = src_abs_img[:-5] + '_'+ str(count).zfill(3) +'.nrrd'
        # dst_abs_gt = src_abs_gt[:-9] + '_'+ str(count).zfill(3) +'_gt.nrrd'
        
        os.rename(src_abs_img,dst_abs_img)
        
        os.rename(src_abs_gt,dst_abs_gt)
        
        
      
if __name__ == '__main__':
    
    main()
