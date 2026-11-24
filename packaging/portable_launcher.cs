using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

static class PortableLauncher {
    // ProcessStartInfo uses a raw Windows command line on .NET Framework.
    static string Quote(string value) {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value) {
            if (c == '\\') { slashes++; continue; }
            result.Append('\\', c == '"' ? slashes * 2 + 1 : slashes).Append(c);
            slashes = 0;
        }
        return result.Append('\\', slashes * 2).Append('"').ToString();
    }

    [STAThread]
    static int Main(string[] args) {
        try {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string app = Path.Combine(root, "_internal");
            var start = new ProcessStartInfo(Path.Combine(app, "python", "pythonw.exe"));
            start.Arguments = "-B " + Quote(Path.Combine(app, "portable_gui.py"));
            foreach (string arg in args) start.Arguments += " " + Quote(arg);
            start.UseShellExecute = false;
            start.EnvironmentVariables["SMART_OVERLAY_NODE"] = Path.Combine(app, "node", "node.exe");
            start.EnvironmentVariables.Remove("NODE_PATH");
            start.EnvironmentVariables.Remove("NODE_OPTIONS");
            using (var child = Process.Start(start)) {
                child.WaitForExit();
                return child.ExitCode;
            }
        } catch (Exception error) {
            MessageBox.Show(error.Message, "Smart Overlay OCR");
            return 1;
        }
    }
}
