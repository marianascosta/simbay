import mujoco


def prepare_model_for_mjx(mj_model):
    """Patch a MuJoCo model for efficient MJX compilation.

    - Zeros unsupported solver options (noslip).
    - Zeros geom margin/gap (unsupported for some collision pairs).
    - Disables collision on arm-link mesh geoms and hand/finger mesh+sphere
      geoms. Only fingertip-pad boxes, the object, and the floor keep
      collision enabled. This dramatically reduces MJX compile time.
    """
    mj_model.opt.noslip_iterations = 0
    mj_model.geom_margin[:] = 0.0
    mj_model.geom_gap[:] = 0.0

    # Disable collision on mesh and sphere geoms (arm links, hand shell,
    # finger meshes).  Keep: plane (floor), box (fingertip pads + object).
    for i in range(mj_model.ngeom):
        gtype = mj_model.geom_type[i]
        if gtype in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SPHERE):
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0

    return mj_model


def batch_mjx_model(mjx_model, masses, body_id, n):
    """Stack N copies of an MJX model and set per-particle body masses.

    Args:
        mjx_model: A single MJX model (from ``mjx.put_model``).
        masses: 1-D JAX array of shape ``(n,)`` with per-particle masses.
        body_id: Integer index of the body whose mass varies.
        n: Number of particles.

    Returns:
        A batched MJX model pytree with a leading dimension of *n*.
    """
    import jax
    import jax.numpy as jnp

    batched = jax.tree.map(lambda x: jnp.stack([x] * n), mjx_model)
    new_body_mass = batched.body_mass.at[:, body_id].set(masses)
    return batched.replace(body_mass=new_body_mass)


def batch_mjx_data(mjx_data, n):
    """Stack N copies of MJX data into a batched pytree."""
    import jax
    import jax.numpy as jnp

    return jax.tree.map(lambda x: jnp.stack([x] * n), mjx_data)
