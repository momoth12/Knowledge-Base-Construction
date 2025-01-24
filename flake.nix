{
  description = "Knowledge Base Construction";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      # Supported systems
      systems = [
        "aarch64-linux"
        "i686-linux"
        "x86_64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShell = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
            overlays = [
              (final: prev: {
                python312 = prev.python312.override {
                  packageOverrides = finalPy: prevPy: {
                    # Fix sentence-transformers
                    sentence-transformers = prevPy.sentence-transformers.overridePythonAttrs (old: {
                      dependencies = old.dependencies ++ [ finalPy.pillow ];
                    });
                  };
                };
              })
            ];
          };
        in
        pkgs.mkShell {
          buildInputs = with pkgs; [
            # For Numpy, Torch, etc.
            stdenv.cc.cc
            zlib

            # Plotting with GTK backend
            gtk3
            gobject-introspection

            # GTK SVG image support
            librsvg
          ];

          packages = with pkgs; [
            (python312.withPackages (
              ps: with ps; [
                # Deep learning libraries
                pandas
                torch
                transformers
                tqdm
                requests
                numpy
                loguru
                pyyaml
                accelerate

                bitsandbytes
                sentencepiece
                matplotlib
                pygobject3
                sentence-transformers
              ]
            ))
          ];

          MPLBACKEND = "GTK3Agg";
        }
      );
    };
}
