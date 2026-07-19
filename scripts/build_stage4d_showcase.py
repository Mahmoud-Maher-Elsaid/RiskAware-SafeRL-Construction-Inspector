from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(sys.argv[1]).resolve()

WORLD_PATH = PROJECT_ROOT / "webots" / "worlds" / "construction_site_stage4d_showcase.wbt"


def vector(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.5f}".rstrip("0").rstrip(".") for value in values)


def appearance(
    color: tuple[float, float, float],
    *,
    roughness: float = 0.8,
    metalness: float = 0.0,
    transparency: float = 0.0,
    emissive: tuple[float, float, float] | None = None,
) -> str:
    fields = [
        f"baseColor {vector(color)}",
        f"roughness {roughness}",
        f"metalness {metalness}",
    ]

    if transparency > 0.0:
        fields.append(f"transparency {transparency}")

    if emissive is not None:
        fields.append(f"emissiveColor {vector(emissive)}")

    joined = "\n          ".join(fields)

    return f"PBRAppearance {{\n          {joined}\n        }}"


def solid_box(
    name: str,
    translation: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    rotation: tuple[float, float, float, float] | None = None,
    roughness: float = 0.8,
    metalness: float = 0.0,
    transparency: float = 0.0,
) -> str:
    rotation_line = ""

    if rotation is not None:
        rotation_line = f"  rotation {vector(rotation)}\n"

    return f"""Solid {{
  translation {vector(translation)}
{rotation_line}  name "{name}"
  children [
    Shape {{
      appearance {
        appearance(
            color,
            roughness=roughness,
            metalness=metalness,
            transparency=transparency,
        )
    }
      geometry Box {{
        size {vector(size)}
      }}
    }}
  ]
  boundingObject Box {{
    size {vector(size)}
  }}
  locked TRUE
}}"""


def visual_box(
    translation: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    rotation: tuple[float, float, float, float] | None = None,
    transparency: float = 0.0,
    emissive: tuple[float, float, float] | None = None,
) -> str:
    rotation_line = ""

    if rotation is not None:
        rotation_line = f"  rotation {vector(rotation)}\n"

    return f"""Pose {{
  translation {vector(translation)}
{rotation_line}  children [
    Shape {{
      appearance {
        appearance(
            color,
            transparency=transparency,
            emissive=emissive,
        )
    }
      geometry Box {{
        size {vector(size)}
      }}
    }}
  ]
}}"""


def solid_cylinder(
    name: str,
    translation: tuple[float, float, float],
    radius: float,
    height: float,
    color: tuple[float, float, float],
    *,
    rotation: tuple[float, float, float, float] | None = None,
    roughness: float = 0.7,
    metalness: float = 0.0,
) -> str:
    rotation_line = ""

    if rotation is not None:
        rotation_line = f"  rotation {vector(rotation)}\n"

    return f"""Solid {{
  translation {vector(translation)}
{rotation_line}  name "{name}"
  children [
    Shape {{
      appearance {
        appearance(
            color,
            roughness=roughness,
            metalness=metalness,
        )
    }
      geometry Cylinder {{
        height {height}
        radius {radius}
        subdivision 24
      }}
    }}
  ]
  boundingObject Cylinder {{
    height {height}
    radius {radius}
    subdivision 16
  }}
  locked TRUE
}}"""


def safety_cone(
    index: int,
    x: float,
    z: float,
) -> str:
    return f"""Pose {{
  translation {x} 0 {z}
  children [
    Pose {{
      translation 0 0.035 0
      children [
        Shape {{
          appearance {appearance((0.08, 0.08, 0.08))}
          geometry Box {{
            size 0.42 0.07 0.42
          }}
        }}
      ]
    }}
    Pose {{
      translation 0 0.34 0
      children [
        Shape {{
          appearance {appearance((1.0, 0.32, 0.02))}
          geometry Cone {{
            bottomRadius 0.18
            height 0.62
            subdivision 24
          }}
        }}
      ]
    }}
    Pose {{
      translation 0 0.28 0
      children [
        Shape {{
          appearance {appearance((0.95, 0.95, 0.95))}
          geometry Cylinder {{
            radius 0.145
            height 0.09
            subdivision 24
          }}
        }}
      ]
    }}
  ]
}}"""


def worker(
    name: str,
    x: float,
    z: float,
    vest_color: tuple[float, float, float],
    helmet_color: tuple[float, float, float],
) -> str:
    skin = (0.72, 0.48, 0.32)
    dark = (0.08, 0.1, 0.14)

    return f"""Pose {{
  translation {x} 0 {z}
  children [
    Pose {{
      translation -0.1 0.42 0
      children [
        Shape {{
          appearance {appearance(dark)}
          geometry Cylinder {{
            radius 0.075
            height 0.82
            subdivision 20
          }}
        }}
      ]
    }}
    Pose {{
      translation 0.1 0.42 0
      children [
        Shape {{
          appearance {appearance(dark)}
          geometry Cylinder {{
            radius 0.075
            height 0.82
            subdivision 20
          }}
        }}
      ]
    }}
    Pose {{
      translation 0 1.05 0
      children [
        Shape {{
          appearance {appearance(vest_color)}
          geometry Cylinder {{
            radius 0.24
            height 0.75
            subdivision 24
          }}
        }}
      ]
    }}
    Pose {{
      translation -0.34 1.04 0
      rotation 0 0 1 -0.18
      children [
        Shape {{
          appearance {appearance(vest_color)}
          geometry Cylinder {{
            radius 0.065
            height 0.68
            subdivision 16
          }}
        }}
      ]
    }}
    Pose {{
      translation 0.34 1.04 0
      rotation 0 0 1 0.18
      children [
        Shape {{
          appearance {appearance(vest_color)}
          geometry Cylinder {{
            radius 0.065
            height 0.68
            subdivision 16
          }}
        }}
      ]
    }}
    Pose {{
      translation 0 1.58 0
      children [
        Shape {{
          appearance {appearance(skin)}
          geometry Sphere {{
            radius 0.2
            subdivision 3
          }}
        }}
      ]
    }}
    Pose {{
      translation 0 1.76 0
      children [
        Shape {{
          appearance {appearance(helmet_color)}
          geometry Cylinder {{
            radius 0.235
            height 0.13
            subdivision 24
          }}
        }}
      ]
    }}
    Pose {{
      translation 0.12 1.69 0
      children [
        Shape {{
          appearance {appearance(helmet_color)}
          geometry Box {{
            size 0.3 0.055 0.42
          }}
        }}
      ]
    }}
  ]
}}"""


world: list[str] = [
    """#VRML_SIM R2025a utf8

WorldInfo {
  coordinateSystem "EUN"
  basicTimeStep 32
  title "RiskAware SafeRL Professional Construction Showcase"
}

Viewpoint {
  orientation -0.57735 0.57735 0.57735 2.0944
  position 15 12 15
  fieldOfView 0.78
}

Background {
  skyColor [
    0.63 0.76 0.91
  ]
}

Fog {
  color 0.78 0.84 0.91
  visibilityRange 70
}

DirectionalLight {
  ambientIntensity 0.62
  direction -0.4 -1 -0.28
  intensity 1.25
  castShadows TRUE
}

DirectionalLight {
  ambientIntensity 0.15
  direction 0.45 -0.7 0.35
  intensity 0.35
  castShadows FALSE
}
"""
]

world.append(
    solid_box(
        "main reinforced concrete construction slab",
        (0.0, -0.12, 0.0),
        (24.0, 0.24, 18.0),
        (0.33, 0.35, 0.38),
        roughness=0.95,
    )
)

world.append(
    visual_box(
        (0.0, 0.012, -5.4),
        (20.5, 0.024, 2.3),
        (0.13, 0.15, 0.18),
    )
)

for x in (
    -8.8,
    -7.2,
    -5.6,
    -4.0,
    -2.4,
    -0.8,
    0.8,
    2.4,
    4.0,
    5.6,
    7.2,
    8.8,
):
    world.append(
        visual_box(
            (x, 0.03, -5.4),
            (0.75, 0.05, 0.08),
            (1.0, 0.78, 0.05),
            emissive=(0.08, 0.05, 0.0),
        )
    )

world.append(
    visual_box(
        (-7.3, 0.018, -1.7),
        (3.2, 0.035, 2.6),
        (0.05, 0.35, 0.82),
        transparency=0.2,
    )
)

world.append(
    visual_box(
        (-5.5, 0.02, 3.4),
        (4.6, 0.04, 4.0),
        (0.74, 0.05, 0.04),
        transparency=0.38,
    )
)

world.append(
    visual_box(
        (5.9, 0.02, 3.7),
        (3.8, 0.04, 2.8),
        (0.95, 0.56, 0.03),
        transparency=0.28,
    )
)

for x in range(-11, 12, 2):
    world.append(
        solid_box(
            f"north fence post {x}",
            (float(x), 0.75, 8.7),
            (0.1, 1.5, 0.1),
            (0.1, 0.12, 0.15),
            metalness=0.65,
        )
    )

    world.append(
        solid_box(
            f"south fence post {x}",
            (float(x), 0.75, -8.7),
            (0.1, 1.5, 0.1),
            (0.1, 0.12, 0.15),
            metalness=0.65,
        )
    )

for z in range(-7, 8, 2):
    world.append(
        solid_box(
            f"east fence post {z}",
            (11.7, 0.75, float(z)),
            (0.1, 1.5, 0.1),
            (0.1, 0.12, 0.15),
            metalness=0.65,
        )
    )

    world.append(
        solid_box(
            f"west fence post {z}",
            (-11.7, 0.75, float(z)),
            (0.1, 1.5, 0.1),
            (0.1, 0.12, 0.15),
            metalness=0.65,
        )
    )

for y in (0.45, 1.05):
    world.append(
        solid_box(
            f"north fence rail {y}",
            (0.0, y, 8.7),
            (23.4, 0.08, 0.08),
            (0.14, 0.16, 0.19),
            metalness=0.7,
        )
    )

    world.append(
        solid_box(
            f"south fence rail {y}",
            (0.0, y, -8.7),
            (23.4, 0.08, 0.08),
            (0.14, 0.16, 0.19),
            metalness=0.7,
        )
    )

    world.append(
        solid_box(
            f"east fence rail {y}",
            (11.7, y, 0.0),
            (0.08, 0.08, 17.4),
            (0.14, 0.16, 0.19),
            metalness=0.7,
        )
    )

    world.append(
        solid_box(
            f"west fence rail {y}",
            (-11.7, y, 0.0),
            (0.08, 0.08, 17.4),
            (0.14, 0.16, 0.19),
            metalness=0.7,
        )
    )

world.append(
    solid_box(
        "building foundation",
        (0.6, 0.12, 3.0),
        (5.8, 0.24, 4.7),
        (0.54, 0.55, 0.57),
    )
)

column_positions = (
    (-1.8, 1.7),
    (0.6, 1.7),
    (3.0, 1.7),
    (-1.8, 4.3),
    (0.6, 4.3),
    (3.0, 4.3),
)

for index, (x, z) in enumerate(column_positions):
    world.append(
        solid_box(
            f"reinforced concrete column {index}",
            (x, 1.9, z),
            (0.38, 3.6, 0.38),
            (0.62, 0.63, 0.64),
        )
    )

for index, z in enumerate((1.7, 4.3)):
    world.append(
        solid_box(
            f"long concrete beam {index}",
            (0.6, 3.55, z),
            (5.2, 0.42, 0.42),
            (0.58, 0.59, 0.61),
        )
    )

for index, x in enumerate((-1.8, 0.6, 3.0)):
    world.append(
        solid_box(
            f"cross concrete beam {index}",
            (x, 3.55, 3.0),
            (0.42, 0.42, 3.0),
            (0.58, 0.59, 0.61),
        )
    )

scaffold_color = (0.82, 0.52, 0.08)

for x in (-2.45, 3.65):
    for z in (1.05, 4.95):
        world.append(
            solid_box(
                f"scaffold vertical {x} {z}",
                (x, 2.0, z),
                (0.09, 4.0, 0.09),
                scaffold_color,
                metalness=0.55,
            )
        )

for y in (0.65, 1.7, 2.75, 3.8):
    for z in (1.05, 4.95):
        world.append(
            solid_box(
                f"scaffold long rail {y} {z}",
                (0.6, y, z),
                (6.2, 0.08, 0.08),
                scaffold_color,
                metalness=0.55,
            )
        )

    for x in (-2.45, 3.65):
        world.append(
            solid_box(
                f"scaffold side rail {y} {x}",
                (x, y, 3.0),
                (0.08, 0.08, 4.0),
                scaffold_color,
                metalness=0.55,
            )
        )

for index, y in enumerate((1.45, 2.55, 3.65)):
    world.append(
        solid_box(
            f"scaffold platform {index}",
            (0.6, y, 4.8),
            (5.9, 0.12, 0.62),
            (0.48, 0.27, 0.09),
        )
    )

world.append(
    solid_box(
        "excavation pit bottom",
        (-5.5, -0.03, 3.4),
        (3.9, 0.06, 3.3),
        (0.13, 0.08, 0.045),
    )
)

for index, (x, z, sx, sz) in enumerate(
    (
        (-5.5, 1.55, 4.5, 0.18),
        (-5.5, 5.25, 4.5, 0.18),
        (-7.85, 3.4, 0.18, 3.9),
        (-3.15, 3.4, 0.18, 3.9),
    )
):
    world.append(
        solid_box(
            f"excavation barrier {index}",
            (x, 0.45, z),
            (sx, 0.9, sz),
            (0.95, 0.45, 0.02),
        )
    )

for index, (x, z) in enumerate(
    (
        (-7.5, 1.25),
        (-6.3, 1.25),
        (-5.1, 1.25),
        (-3.9, 1.25),
        (-7.5, 5.55),
        (-6.3, 5.55),
        (-5.1, 5.55),
        (-3.9, 5.55),
    )
):
    world.append(
        safety_cone(
            index,
            x,
            z,
        )
    )

world.append(
    solid_box(
        "site office container",
        (7.5, 1.25, 4.9),
        (5.0, 2.5, 2.8),
        (0.11, 0.42, 0.58),
        metalness=0.25,
    )
)

world.append(
    visual_box(
        (7.5, 1.3, 3.47),
        (1.15, 1.65, 0.04),
        (0.08, 0.09, 0.11),
    )
)

world.append(
    visual_box(
        (6.1, 1.5, 3.47),
        (1.2, 0.85, 0.04),
        (0.35, 0.72, 0.92),
        emissive=(0.03, 0.08, 0.11),
    )
)

world.append(
    visual_box(
        (8.65, 1.5, 3.47),
        (1.2, 0.85, 0.04),
        (0.35, 0.72, 0.92),
        emissive=(0.03, 0.08, 0.11),
    )
)

world.append(
    solid_box(
        "tower crane concrete base",
        (8.0, 0.35, -3.0),
        (2.1, 0.7, 2.1),
        (0.52, 0.53, 0.55),
    )
)

world.append(
    solid_box(
        "tower crane mast",
        (8.0, 3.5, -3.0),
        (0.72, 6.4, 0.72),
        (0.93, 0.62, 0.02),
        metalness=0.35,
    )
)

world.append(
    solid_box(
        "tower crane main jib",
        (5.3, 6.45, -3.0),
        (6.2, 0.34, 0.34),
        (0.93, 0.62, 0.02),
        metalness=0.35,
    )
)

world.append(
    solid_box(
        "tower crane counter jib",
        (9.5, 6.45, -3.0),
        (2.4, 0.34, 0.34),
        (0.93, 0.62, 0.02),
        metalness=0.35,
    )
)

world.append(
    solid_box(
        "tower crane hook cable",
        (3.0, 4.65, -3.0),
        (0.045, 3.3, 0.045),
        (0.08, 0.08, 0.09),
        metalness=0.8,
    )
)

world.append(
    solid_box(
        "tower crane hook",
        (3.0, 2.95, -3.0),
        (0.26, 0.28, 0.26),
        (0.12, 0.12, 0.13),
        metalness=0.8,
    )
)

for pallet_index, x in enumerate((2.8, 4.1, 5.4)):
    world.append(
        solid_box(
            f"timber pallet {pallet_index}",
            (x, 0.12, -1.25),
            (1.0, 0.24, 1.1),
            (0.42, 0.23, 0.08),
        )
    )

    for layer in range(4):
        world.append(
            solid_box(
                f"timber stack {pallet_index} {layer}",
                (
                    x,
                    0.34 + layer * 0.16,
                    -1.25,
                ),
                (0.9, 0.12, 0.92),
                (0.58, 0.33, 0.12),
            )
        )

for pipe_index in range(5):
    world.append(
        solid_cylinder(
            f"steel pipe {pipe_index}",
            (
                6.3,
                0.22 + pipe_index * 0.18,
                -0.9,
            ),
            0.13,
            2.6,
            (0.22, 0.25, 0.29),
            rotation=(
                0.0,
                0.0,
                1.0,
                1.5708,
            ),
            metalness=0.75,
        )
    )

world.append(
    worker(
        "worker ppe station",
        -7.2,
        -1.7,
        (0.95, 0.55, 0.03),
        (0.95, 0.82, 0.05),
    )
)

world.append(
    worker(
        "worker scaffold observer",
        4.2,
        2.0,
        (0.95, 0.42, 0.03),
        (0.95, 0.82, 0.05),
    )
)

world.append(
    worker(
        "site safety supervisor",
        6.0,
        4.3,
        (0.12, 0.66, 0.22),
        (0.96, 0.96, 0.96),
    )
)

for checkpoint_index, (x, z) in enumerate(
    (
        (-3.6, -5.4),
        (0.0, -5.4),
        (3.6, -5.4),
        (-2.4, -1.7),
        (4.3, 0.7),
    )
):
    world.append(
        solid_cylinder(
            f"inspection checkpoint {checkpoint_index}",
            (x, 0.09, z),
            0.28,
            0.18,
            (0.02, 0.78, 0.92),
            roughness=0.3,
            metalness=0.15,
        )
    )

world.append(
    """DEF SHOWCASE_ROBOT Robot {
  translation -8.3 0.12 -5.4
  rotation 0 1 0 0
  name "professional construction inspection robot"
  controller "showcase_robot"
  children [
    GPS {
      name "gps"
    }
    InertialUnit {
      name "inertial unit"
    }
    Compass {
      name "compass"
    }
    Camera {
      translation 0.22 0.46 0
      rotation 0 1 0 -1.5708
      name "inspection camera"
      fieldOfView 1.05
      width 320
      height 240
    }
    Pose {
      translation 0 0.18 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.025 0.25 0.65
            roughness 0.32
            metalness 0.35
          }
          geometry Box {
            size 0.62 0.26 0.46
          }
        }
      ]
    }
    Pose {
      translation 0.12 0.34 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.03 0.09 0.2
            roughness 0.25
            metalness 0.45
          }
          geometry Box {
            size 0.34 0.16 0.32
          }
        }
      ]
    }
    Pose {
      translation 0.19 0.46 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 1 0.55 0.02
            emissiveColor 0.12 0.04 0
            roughness 0.25
            metalness 0.15
          }
          geometry Box {
            size 0.19 0.09 0.2
          }
        }
      ]
    }
    Pose {
      translation 0.31 0.46 0
      rotation 0 0 1 1.5708
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.02 0.03 0.04
            roughness 0.15
            metalness 0.35
          }
          geometry Cylinder {
            radius 0.07
            height 0.08
            subdivision 24
          }
        }
      ]
    }
    Pose {
      translation -0.21 0.25 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.86 0.88 0.9
            roughness 0.4
            metalness 0.65
          }
          geometry Box {
            size 0.08 0.5 0.08
          }
        }
      ]
    }
    Pose {
      translation -0.21 0.52 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 1 0.53 0.02
            emissiveColor 0.2 0.05 0
            roughness 0.25
            metalness 0.1
          }
          geometry Sphere {
            radius 0.075
            subdivision 3
          }
        }
      ]
    }
    HingeJoint {
      jointParameters HingeJointParameters {
        axis 0 0 1
        anchor 0 0 0.245
      }
      device [
        RotationalMotor {
          name "left wheel motor"
          maxVelocity 12
          maxTorque 2.5
        }
        PositionSensor {
          name "left wheel sensor"
        }
      ]
      endPoint Solid {
        translation 0 0 0.245
        rotation 1 0 0 1.5708
        name "left showcase wheel"
        children [
          Shape {
            appearance PBRAppearance {
              baseColor 0.025 0.025 0.03
              roughness 0.96
              metalness 0
            }
            geometry Cylinder {
              height 0.075
              radius 0.12
              subdivision 32
            }
          }
        ]
        boundingObject Cylinder {
          height 0.075
          radius 0.12
          subdivision 20
        }
        physics Physics {
          density -1
          mass 0.32
        }
      }
    }
    HingeJoint {
      jointParameters HingeJointParameters {
        axis 0 0 1
        anchor 0 0 -0.245
      }
      device [
        RotationalMotor {
          name "right wheel motor"
          maxVelocity 12
          maxTorque 2.5
        }
        PositionSensor {
          name "right wheel sensor"
        }
      ]
      endPoint Solid {
        translation 0 0 -0.245
        rotation 1 0 0 1.5708
        name "right showcase wheel"
        children [
          Shape {
            appearance PBRAppearance {
              baseColor 0.025 0.025 0.03
              roughness 0.96
              metalness 0
            }
            geometry Cylinder {
              height 0.075
              radius 0.12
              subdivision 32
            }
          }
        ]
        boundingObject Cylinder {
          height 0.075
          radius 0.12
          subdivision 20
        }
        physics Physics {
          density -1
          mass 0.32
        }
      }
    }
    Pose {
      translation -0.25 -0.015 0
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.08 0.08 0.09
            roughness 0.5
            metalness 0.4
          }
          geometry Sphere {
            radius 0.065
            subdivision 3
          }
        }
      ]
    }
  ]
  boundingObject Group {
    children [
      Pose {
        translation 0 0.18 0
        children [
          Box {
            size 0.62 0.26 0.46
          }
        ]
      }
      Pose {
        translation -0.25 -0.015 0
        children [
          Sphere {
            radius 0.065
            subdivision 2
          }
        ]
      }
    ]
  }
  physics Physics {
    density -1
    mass 4.5
  }
}

Robot {
  name "professional showcase supervisor"
  supervisor TRUE
  controller "showcase_supervisor"
}
"""
)

WORLD_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

WORLD_PATH.write_text(
    "\n\n".join(world).rstrip() + "\n",
    encoding="utf-8",
    newline="\n",
)

print("Created Stage 4D showcase world:")
print(WORLD_PATH)
print(f"World size: {WORLD_PATH.stat().st_size} bytes")
