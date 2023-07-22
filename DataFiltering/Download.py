import os
import requests
import pandas as pd
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError
import concurrent.futures
import socket
import argparse
import time
from tqdm import tqdm

def download_dataset(root_folder_read, root_folder_write, max_imgs):
    # Download images given a pytorch folder of urls given in jsons
    master_df = pd.DataFrame(columns=['URL', 'Path', 'TEXT'])
    for class_name in os.listdir(root_folder_read):
        class_folder = os.path.join(root_folder_read, class_name.replace('/', 'or'))
        write_folder = os.path.join(root_folder_write, class_name.replace('/', 'or'))
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)
        for chunks in os.listdir(class_folder):
            chunk_path = os.path.join(class_folder, chunks)
            df = pd.read_json(chunk_path)
            folder_path = os.path.join(root_folder_write, class_name)
            df_out = download_url_list(root_folder_write, df, class_name, max_imgs)
            master_df = pd.concat([master_df, df_out]) 
            master_df.to_csv(root_folder_write + '.csv')


def download_image(url, filename, df, urls, paths, captions, sims, j):
    #texts = caption
    retry_strategy = Retry(
        total=1,
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Create a requests session with the retry mechanism
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter) # (connect timeout, read timeout)
    #print(url)
    # Set the socket timeout for the session
    socket.setdefaulttimeout(1)
    # Make a request using the session
    try:
        if not(os.path.exists(filename)):
            response = session.get(url, timeout=1)
            if response.status_code == 200 and response.history == [] and response.url == url:
                with open(filename, 'wb') as f:
                    f.write(requests.get(url).content)
                print(f'Downloaded {url}')
                urls.append(url)
                paths.append(filename)
                captions.append(df.TEXT[j])
                sims.append(df.similarity[j])
            else:
                print(f'Failed to download {url} (status code: {response.status_code})')
        else:
            print('Already downloaded')
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while downloading {url}")
    except SSLError:
        print(f"SSL error occurred while downloading {url}")
    except Exception as e:
        print(f"Error occurred while downloading {url}: {str(e)}")

def download_url_list(root_folder, df, class_name, max_imgs):
    start_time = time.time()
    image_urls = df.URL
    texts = df.TEXT
    # Create a directory to save the downloaded images
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    folder_path = os.path.join(root_folder, class_name)
    print(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a thread pool to download the images
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
        # Loop through the image URLs and download each safe image
        urls = []
        paths = []
        captions = []
        sims = []
        futures = []
        for j, url in tqdm(enumerate(image_urls[:max_imgs])):
            if 'png' or 'jpg' in url:
                filename = os.path.join(folder_path, os.path.basename(url))
                future = executor.submit(download_image, url, filename, df, urls, paths, captions, sims, j)
                futures
    print(f'Downloaded {len(urls)} out of {len(image_urls[:max_imgs])} images.')
    print(print("Downloaded in --- %s seconds ---" % (time.time() - start_time)))
    print('Images per second: {}'.format(len(urls)/(time.time() - start_time)))

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description=""
    )

    # Add arguments
    parser.add_argument("--r", type=str, help="Path to SQLite database file.")
    parser.add_argument("--w", type=str, help="")
    parser.add_argument("--n", type=int, default=1000)

    # Parse arguments
    args = parser.parse_args()

    # Call main function with arguments
    download_dataset(args.r, args.w, args.n)