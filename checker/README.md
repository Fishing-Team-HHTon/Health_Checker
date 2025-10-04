## Запуск программы

### Установка и найстройка зависимостей:
```bash
winget install -e --id MSYS2.MSYS2

$msys = "C:\msys64\usr\bin\bash.exe"
& $msys -lc "pacman -Sy --noconfirm --needed mingw-w64-x86_64-toolchain"

$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"

where dlltool
where gcc

rustup default stable-x86_64-pc-windows-gnu
cargo clean
```

### Запуск проекта:
```bash
    cargo run --release -- --port COM4 --fs 100 --vref 5.0 --hp --ma 3 --echo-mode-lines --mode ecg
```