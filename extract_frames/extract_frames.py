from pathlib import Path
import subprocess
import shutil

outfps = 2

scriptdir = Path(__file__).resolve().parent

inpaths = scriptdir.glob("*.mkv")

for i, path in enumerate(inpaths):
    outdir = path.parent / path.stem
    if outdir.is_dir():
        shutil.rmtree(outdir)
    elif outdir.is_file():
        outdir.unlink()
    
    outdir.mkdir()
    
    subprocess.run([
        "ffmpeg",
        "-i", str(path),
        "-r", str(outfps),
        str(outdir / "frame-{}-%04d.png".format(i))
    ])