:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

member

.. auto{{ objtype }}:: {{ fullname | replace("fedsim.", "fedsim::") }}

{# In the fullname, the module name is ambiguous. Using a `::` separator 
specifies `fedsim` as the module name. #}
