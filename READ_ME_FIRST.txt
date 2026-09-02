GA Clinic for macOS — offline
=============================

1. Double-click  oct_ga_clinic_mac.tar.gz  to unpack it (creates the  oct_ga_clinic_mac  folder).

2. Clear the download quarantine ONCE so macOS will run the bundled files:
     - Open Terminal (Spotlight -> "Terminal").
     - Type:   xattr -dr com.apple.quarantine     (note the trailing space)
     - Drag the unpacked  oct_ga_clinic_mac  folder onto the Terminal window, then press Return.

3. Double-click  run.command  inside the folder.
     - First launch only: it unpacks Python and installs its libraries (a couple of minutes, NO internet
       needed), then your browser opens at  http://127.0.0.1:8021/ .
     - Double-clicking again while it is running simply reopens the browser page.
     - If macOS still blocks it: right-click run.command -> Open -> Open (once).

4. Database = the patients tracked on this Mac. Upload E2E = Browse for a .E2E file (or paste its path),
   choose a scan, then the Bruch's-membrane source. Export Excel saves the database as .xlsx.
   Patient state, managed drag/drop copies and caches live in user_data/. To stop, close the Terminal window.

Everything runs locally and offline; nothing is installed system-wide.
