/*
 * Utter.app's main executable.
 *
 * Why a compiled stub rather than a shell script, and why exec rather than
 * spawn: macOS attributes Accessibility and Microphone permission to the
 * *bundle* whose main executable was launched. A shell script in
 * Contents/MacOS works for launching but leaves the running process looking
 * like /bin/sh to some of the machinery, and the author has already lost an
 * evening to a permission that was granted to the terminal instead of the
 * app.
 *
 * execv keeps the process id and the launch context, so the Python that ends
 * up running is still the process macOS started from the bundle. Verified by
 * reading NSBundle.mainBundle().bundleIdentifier() from inside it — see
 * `utter bundle --check`.
 *
 * The interpreter path is compiled in rather than searched for. A tool that
 * silently picks up a different Python than the one its dependencies are
 * installed into fails in a way that takes an hour to understand.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <libgen.h>
#include <mach-o/dyld.h>

#ifndef UTTER_PYTHON
#error "compile with -DUTTER_PYTHON=\"/path/to/venv/bin/python\""
#endif

int main(int argc, char *argv[]) {
    /* Contents/MacOS/Utter -> Contents/Resources */
    char self[4096];
    uint32_t size = sizeof(self);
    if (_NSGetExecutablePath(self, &size) != 0) {
        fprintf(stderr, "Utter: cannot locate my own bundle\n");
        return 1;
    }
    char *macos_dir = dirname(self);            /* .../Contents/MacOS   */
    char contents[4096];
    snprintf(contents, sizeof(contents), "%s", dirname(macos_dir));

    char resources[4096];
    snprintf(resources, sizeof(resources), "%s/Resources", contents);
    setenv("UTTER_APP_RESOURCES", resources, 1);

    /* The load-bearing line.
     *
     * execv replaces this image with Python's, and CoreFoundation then locates
     * "the main bundle" by walking up from the *running executable* — which is
     * now ~/.venvs/utter/bin/python, outside the bundle entirely. NSBundle
     * came back nil, and a nil bundle means macOS files the Accessibility and
     * Microphone grants against Python rather than against Utter: exactly the
     * mistake that put this project's first grant on Terminal.
     *
     * CFProcessPath tells CoreFoundation where the process "is" regardless of
     * what it is running. Undocumented but long-standing, and it is how every
     * wrapper-style .app solves this. Verified with `Utter.app/.../Utter
     * --check`, which prints the bundle identifier it ends up with. */
    setenv("CFProcessPath", self, 1);

    /* Unbuffered, so the log file is useful while the app is still running
     * rather than only after it exits. */
    setenv("PYTHONUNBUFFERED", "1", 1);

    char *args[8];
    int n = 0;
    args[n++] = (char *)UTTER_PYTHON;
    args[n++] = "-m";
    args[n++] = "backend.app";
    for (int i = 1; i < argc && n < 7; i++) {
        args[n++] = argv[i];
    }
    args[n] = NULL;

    execv(UTTER_PYTHON, args);

    /* Only reached if execv failed. A GUI app that dies silently is the worst
     * possible outcome, so say why somewhere the user can find it. */
    fprintf(stderr, "Utter: could not start %s: %s\n", UTTER_PYTHON, strerror(errno));
    return 1;
}
