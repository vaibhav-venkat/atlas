The central architecture should stay:

```text
Atlas
semantic scientific types
      │
      ▼
Raven tensors / Nx operations
      │
      ▼
Rune transformations
      │
      ▼
Atlas-aware derivatives
```

The rule throughout development should be:

> Atlas owns scientific meaning. Raven owns numerical representation and execution. Rune owns program transformations and AD.

---

# Phase 0 — Establish the semantic boundary

Before implementing many features, determine exactly what information belongs in Atlas versus Raven.

```text
Atlas
├── units
├── frames
├── geometric roles
├── named axes
└── semantic properties

Raven
├── dtype
├── device
├── storage
├── shape
├── strides/layout
├── BLAS/tensor operations
└── execution

Rune
├── grad
├── jvp
├── vjp
├── jacobian
└── transformations
```

The important architectural constraint:

```text
Atlas tensor
     │
     ▼
Raven tensor

should ideally have
the same runtime representation
```

Avoid designing Atlas around runtime records like:

```ocaml
type atlas_tensor = {
  tensor : Nx.t;
  unit : unit_description;
  frame : frame_description;
  ...
}
```

unless metadata is genuinely required at runtime.

Prefer static information:

```ocaml
('unit_, 'frame, 'kind, 'axes, 'data) Atlas.t
```

with most of it erased.

### Phase 0 questions to settle

You don't need perfect answers, but establish conventions for:

```text
Are units represented by phantom types or modules?

Are coordinate frames generative modules?

Are point/vector distinctions GADTs?

How much shape information does Atlas duplicate from Raven?

How does Atlas expose raw Raven tensors?

How does Rune see through Atlas abstractions?

What semantic information survives AD?
```

### Exit condition

You can explain clearly why:

```ocaml
Atlas.unwrap x
```

is essentially free and what information is compile-time-only.

---

# Phase 1 — Minimal Atlas: units + scalar/tensor quantities

Start with the simplest universally useful semantic invariant:

## Physical dimensions

Do **not** begin with every SI unit.

Start with dimension algebra:

```text
length
time
mass
temperature
...
```

and derived dimensions:

```text
velocity     = length / time
acceleration = length / time²
force        = mass * length / time²
energy       = force * length
```

The user-facing API should feel straightforward:

```ocaml
let distance =
  Atlas.Quantity.metres 10.

let duration =
  Atlas.Quantity.seconds 2.

let velocity =
  Atlas.div distance duration
```

Conceptually:

```text
velocity :
  metre / second
```

and:

```ocaml
Atlas.add distance duration
```

must fail to type-check.

---

## Immediately support Raven tensors

Don't let Atlas spend long as a scalar-only package.

Very early:

```ocaml
let position :
  Atlas.metre Atlas.tensor =
  Atlas.of_raven x
```

Then basic Raven-like operations:

```text
add
sub
mul
div
sum
mean
reshape
broadcast
matmul
dot
norm
```

The important exercise is deciding how semantic units transform.

For example:

```text
metre × metre
     ↓
metre²
```

and:

```text
joule / metre
     ↓
newton
```

while:

```text
metre + second
```

is rejected.

---

# Phase 1B — Prove zero-cost Raven interoperability

Do this early rather than assuming it will work later.

Benchmark:

```text
Raven:
Nx.add x y

Atlas:
Atlas.add x y
```

and verify:

```text
same storage
same Raven primitive
same generated hot path
negligible/no allocation difference
```

The objective is not necessarily literal zero instructions everywhere, but Atlas shouldn't create a shadow tensor runtime.

Architecture:

```text
Atlas expression
      │
      ▼
static semantic checking
      │
      ▼
Raven operation
      │
      ▼
normal Raven execution
```

### Phase 1 exit condition

A scientist can perform ordinary unit-safe tensor arithmetic without feeling like they're using a separate numerical system.

---

# Phase 2 — Coordinate frames + geometry

Once the Raven integration is solid, add what I think is Atlas's most differentiating capability.

```text
World
Camera
Robot
Sensor
Body
Map
...
```

Use generative types so independently created frames cannot accidentally unify.

Something like:

```ocaml
module World = Atlas.Frame.Make ()
module Camera = Atlas.Frame.Make ()
```

Then:

```ocaml
let p :
  (Atlas.metre, Camera.t, Atlas.point3) Atlas.t =
  Camera.point xyz
```

Now distinguish:

```text
Point
Vector
Covector
Direction
Normal
```

at least where meaningful.

---

## Geometry rules

Atlas should statically encode:

```text
point - point
    ↓
displacement

point + displacement
    ↓
point

displacement + displacement
    ↓
displacement
```

but reject:

```text
World.point + Camera.displacement

World.vector + Camera.vector

point + point
```

unless there is an explicitly defined mathematical interpretation.

---

# Transform API

Transforms become first-class typed objects:

```ocaml
camera_to_world :
  (Camera.t, World.t) Atlas.transform
```

Then:

```ocaml
Atlas.transform camera_to_world camera_point
```

produces:

```text
World point
```

without the caller manually casting semantic information.

Composition should itself be checked:

```text
Camera → Robot
Robot  → World
----------------
Camera → World
```

whereas:

```text
Camera → Robot
Sensor → World
```

cannot compose.

---

# Phase 2 Raven integration

Make sure transforms eventually lower to ordinary Raven primitives:

```text
Atlas transform
      │
      ▼
matmul / affine operation
      │
      ▼
Raven
```

Atlas should not implement its own matrix engine.

### Phase 2 exit condition

You can write a small robotics/geometry example where multiple plausible `float[3]` mistakes become compile-time errors.

That is an important demo because it immediately communicates why Atlas exists.

---

# Phase 3 — Rune integration becomes first-class

This should probably be developed somewhat in parallel with Phase 2 rather than waiting until Atlas is "finished."

The first requirement is simple:

```ocaml
Rune.grad
Rune.jvp
Rune.vjp
```

must work on Atlas-backed Raven computations.

Initially, it is acceptable if Atlas semantics are somewhat conservative.

The architecture:

```text
Atlas function
     │
     ▼
unwrap / lower
     │
     ▼
Rune transformation
     │
     ▼
Raven derivative computation
     │
     ▼
reapply derived Atlas semantics
```

Eventually, that semantic transformation should itself be principled rather than manually annotated.

---

# Phase 3A — Unit-aware differentiation

This is the first strong Atlas/Rune feature.

Suppose:

```text
x : metre
f(x) : joule
```

Then:

```text
df/dx : joule / metre
```

Atlas should derive that.

Likewise:

```text
velocity(x)
position : metre
time     : second

d position / d time
    ↓
metre / second
```

This gives Rune a semantic derivative signature instead of just:

```text
Tensor → Tensor
```

---

# Phase 3B — Tangents and cotangents

This is where Atlas can become unusually strong.

Forward AD naturally deals with tangent values:

```text
x
+
dx
```

Reverse AD deals with cotangent-like quantities.

Atlas should eventually distinguish:

```text
primal
tangent
cotangent
```

rather than treating all three as interchangeable arrays.

Conceptually:

```text
Rune.jvp
    │
    ▼
tangent propagation
```

and:

```text
Rune.vjp
    │
    ▼
cotangent propagation
```

That is particularly valuable for geometry.

A point in a manifold/frame and the tangent to that point should not necessarily be the exact same semantic object.

---

# Phase 3C — Frame-aware differentiation

Example:

```text
E :
World.position → joule
```

Then:

```text
grad E
```

should conceptually produce:

```text
World.covector with units joule/metre
```

not:

```text
Camera.vector
```

and not an anonymous float tensor.

This makes the combination:

```text
Atlas + Rune
```

much more interesting than Atlas alone.

### Phase 3 exit condition

Examples like this work naturally:

```ocaml
let energy position = ...

let force =
  Rune.grad energy position
```

and the compiler knows the resulting semantic quantity.

---

# Phase 4 — Named axes

Once units + frames + Rune work reliably, add tensor semantics.

For example:

```text
[batch; time; xyz]
```

versus:

```text
[time; batch; xyz]
```

Even though both might have:

```text
float[32, 1000, 3]
```

Atlas knows their roles differ.

Possible operations:

```ocaml
Atlas.Axis.sum Time x
Atlas.Axis.mean Batch x
Atlas.Axis.align x y
```

rather than forcing scientists to remember:

```ocaml
Nx.sum ~axis:1 x
```

---

## Named axes + Raven

Atlas should translate:

```text
Axis.Time
```

into the appropriate Raven dimension/index at compilation/API boundary.

```text
semantic axis
     │
     ▼
resolve physical dimension
     │
     ▼
Raven reduction
```

The runtime shouldn't carry large symbolic axis structures unless necessary.

---

## Named axes + Rune

Rune mostly shouldn't care.

If:

```text
x :
[batch; time; feature]
```

then:

```text
grad loss x
```

should retain:

```text
[batch; time; feature]
```

unless the transformation mathematically changes axis structure.

### Phase 4 exit condition

A meaningful neural/scientific tensor program no longer contains unexplained numeric axis indices throughout application code.

---

# Phase 5 — Semantic linear algebra

This is where Atlas starts becoming a genuine scientific linear-algebra language.

Introduce types/properties such as:

```text
scalar
vector
covector
linear_map
matrix
symmetric_operator
orthogonal_transform
SPD_operator
covariance
metric
```

Do this carefully.

Don't make:

```text
every Nx.matrix
```

require an elaborate type.

Instead, allow scientists to progressively refine:

```text
Raven tensor
     ↓
Atlas matrix
     ↓
Atlas symmetric matrix
     ↓
Atlas SPD operator
```

when they actually possess the corresponding evidence.

---

## Example

```ocaml
let covariance =
  Atlas.Covariance.of_samples samples
```

Now Atlas knows more than:

```text
float[n,n]
```

It potentially knows:

```text
linear operator
symmetric
positive semidefinite
units = X²
axes = feature × feature
```

That information can already make Raven APIs safer:

```ocaml
Atlas.Linear.solve ...
Atlas.Linear.quadratic_form ...
```

And Rune can derive better semantic derivative information.

This also lays groundwork for Veritas later, but you don't need Veritas to justify this phase.

---

# Phase 6 — Optional domain semantics

Only once the core abstraction has survived real use should Atlas expand into things such as:

```text
mesh vertices
mesh cells
mesh faces

grid-centered values
edge-centered values

probability/simplex semantics

complex coordinate systems

uncertainty

manifolds
```

I would **not commit now** to which of these belongs in Atlas core.

Instead:

```text
Atlas.Core
├── units
├── frames
├── geometry
├── axes
└── linear algebra semantics

Atlas.Mesh
Atlas.Uncertainty
Atlas.Robotics
...
```

can emerge from actual applications.

That keeps the core from turning into a giant type-level taxonomy of science.

---

# Development should not be strictly sequential

I would think of it more like this:

```text
                         ┌───────────────┐
                         │ Atlas Core    │
                         │ representation│
                         └───────┬───────┘
                                 │
               ┌─────────────────┼────────────────┐
               ▼                 ▼                ▼
             Units             Frames          Raven
               │                 │                │
               └────────┬────────┘                │
                        ▼                         │
                     Geometry                    │
                        │                         │
                        └───────────┬─────────────┘
                                    ▼
                                  Rune
                              semantic AD
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                     Named axes       Semantic linear
                                          algebra
```

So, for example, Rune work should begin **before** units/frames are completely exhaustive.

---

# What I would actually build first

A practical first slice should be small:

```text
ATLAS v0
│
├── semantic wrapper over Raven tensor
│
├── units
│   ├── length
│   ├── time
│   ├── mass
│   └── dimension algebra
│
├── frames
│
├── point / displacement / vector
│
├── transforms
│
├── basic Raven operations
│
└── Rune
    ├── grad
    ├── jvp
    └── vjp
```

Then make one application that exercises all of it.

For example:

```text
             Camera observation
                    │
                    ▼
             Camera.position
                    │
              transform
                    ▼
              World.position
                    │
                    ▼
               potential E
                    │
                 Rune.grad
                    ▼
              World force
```

That one example tests:

```text
units
frames
points
vectors
transforms
Raven execution
Rune differentiation
semantic preservation
```

If that feels clean, the core architecture is probably viable.

---

# What I would explicitly postpone

Don't initially attempt:

```text
full dimensional-analysis universe
arbitrary symbolic units
uncertainty types
mesh semantics
manifolds
PDE semantics
every Raven operation
full compile-time tensor shapes
dependent-type-like dimensions
automatic physical-law checking
```

Those could consume the project before there is a usable library.

Instead:

```text
small semantic core
        ↓
excellent Raven interoperability
        ↓
excellent Rune interoperability
        ↓
real applications
        ↓
expand semantics where pain appears
```

---

# Overall roadmap

```text
PHASE 0
Architecture / representation
      │
      ▼
Atlas semantics erase cleanly to Raven
      │
      ▼
PHASE 1
Units + Raven tensor operations
      │
      ▼
PHASE 2
Frames + points/vectors + transforms
      │
      ├──────────────┐
      │              │
      ▼              ▼
PHASE 3          continuous
Rune AD          Raven coverage
      │
      ▼
unit-aware + frame-aware
tangent/cotangent semantics
      │
      ▼
PHASE 4
Named axes
      │
      ▼
PHASE 5
Semantic linear algebra
      │
      ▼
PHASE 6
Domain extensions driven by users
```

The most important checkpoint is actually **after Phase 3**. At that point Atlas should already have a strong identity:

> **Raven tensors with statically checked scientific semantics that remain meaningful through Rune differentiation.**

If that works well, named axes, semantic linear algebra, meshes, robotics extensions, and other features become incremental growth rather than requirements for Atlas to be useful.