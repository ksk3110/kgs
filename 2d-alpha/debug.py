import lib
import dolfinx.io

domain, cell_tags, facet_tags = lib.rho_fields_from_controls_robust(
    [
        (15.0,  2.0),  (15.0,  8.0), (13.0, 10.0),
        (-13.0,  10.0),  (-15.0, 8.0), (-15.0, 2.0)
    ],
    lc_wall=0.5,
    lc_domain=0.5
)

with dolfinx.io.XDMFFile(domain.comm, "dist/debug.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    if cell_tags is not None:
        xdmf.write_meshtags(cell_tags, domain.geometry)
    if facet_tags is not None:
        xdmf.write_meshtags(facet_tags, domain.geometry)
