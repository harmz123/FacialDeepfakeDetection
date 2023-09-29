import os
from os.path import join
from tqdm import tqdm
import cv2
from PIL import Image, ImageChops,ImageEnhance
# from deepface import DeepFacePY
# import matplotlib.pyplot as plt


def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/

num=1


def extract_sequences(videopath,trainpath,testpath,num_i,num_max):
    numpervid = num
    numvideotrain = 700
    numvideotest = 300
    diffinseq = 10


    downloaded_videos_path = join(videopath)
    images_out_path = trainpath
    frame_num = 0


    for video_id in tqdm(sorted(os.listdir(downloaded_videos_path))):
        video_seq_path = join(downloaded_videos_path, video_id)


        # Open reader
        reader = cv2.VideoCapture(video_seq_path)
        curr_seq = 0
        curr_seq_image_count = 0


        if frame_num == numvideotrain:
            images_out_path = testpath
            numpervid = num
        if frame_num == numvideotrain+numvideotest:
            break



        while reader.isOpened():
            _, image = reader.read()

            if image is None:
                break


            curr_seq_image_count += 1
            if curr_seq_image_count >= diffinseq:


                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(100, 100),
                    maxSize=(300,300)
                )
                if  len(faces)==0:
                    continue



                for (x, y, w, h) in faces:
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = image[y:y + h, x:x + w]

                # image = cv2.cvtColor(roi_color,cv2.COLOR_BGR2RGB)
                image=roi_color


                curr_seq += 1
                out_fn = str(frame_num)+'_'+str(curr_seq)+'.jpg'

                if num_i == num_max - 1:
                    if curr_seq==1:

                        os.makedirs(join(images_out_path, 'real', video_id),exist_ok=True)
                        images_out_path2=join(images_out_path, 'real', video_id)

                    os.makedirs(join(images_out_path2, str(curr_seq)),exist_ok=True)
                    cv2.imwrite(join(images_out_path2, str(curr_seq),out_fn),
                                image)
                    img2=convert_to_ela_image(join(images_out_path2, str(curr_seq),out_fn))
                    img2.save(join(images_out_path2, str(curr_seq),out_fn), 'JPEG', quality=90)

                else:
                    if curr_seq==1:
                        os.makedirs(join(images_out_path, 'fake', video_id),exist_ok=True)
                        images_out_path2=join(images_out_path, 'fake', video_id)
                    os.makedirs(join(images_out_path2,str(curr_seq)),exist_ok=True)
                    cv2.imwrite(join(images_out_path2, str(curr_seq),out_fn),
                                image)
                    img2 = convert_to_ela_image(join(images_out_path2,  str(curr_seq),out_fn))
                    img2.save(join(images_out_path2, str(curr_seq),out_fn), 'JPEG', quality=90)

            if curr_seq == numpervid:
                break

        frame_num += 1

        # Finish reader
        reader.release()


data_path= '/home/anonymous/PycharmProjects/FacialDeepfakeDetection/dataN1'
videosources=['FaceForensics/manipulated_sequences/Deepfakes/c23/videos',
              'FaceForensics/manipulated_sequences/Face2Face/c23/videos',
              'FaceForensics/manipulated_sequences/FaceSwap/c23/videos',
              'FaceForensics/manipulated_sequences/NeuralTextures/c23/videos',
              'FaceForensics/original_sequences/youtube/c23/videos']
traindests=['FaceForensics_C23_Images_Train/Manipulated/Deepfakes',
            'FaceForensics_C23_Images_Train/Manipulated/Face2Face',
            'FaceForensics_C23_Images_Train/Manipulated/FaceSwap',
            'FaceForensics_C23_Images_Train/Manipulated/NeuralTextures',
            'FaceForensics_C23_Images_Train/Pristine']
testdests=['FaceForensics_C23_Images_Test/Manipulated/Deepfakes',
           'FaceForensics_C23_Images_Test/Manipulated/Face2Face',
           'FaceForensics_C23_Images_Test/Manipulated/FaceSwap',
           'FaceForensics_C23_Images_Test/Manipulated/NeuralTextures',
           'FaceForensics_C23_Images_Test/Pristine']




n=len(videosources)

for i in range(0,n):
    extract_sequences(join(data_path,videosources[i]),join(data_path,traindests[i]),join(data_path,testdests[i]),i,n)
