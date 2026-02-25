export default async () => {
  const upstream = "https://aam.earthrotation.net/aam_fcs_ser/aam_geos_fcs.txt";

  try {
    const r = await fetch(upstream, {
      headers: { "User-Agent": "phillyweatherauthority-aam-proxy" }
    });

    if (!r.ok) {
      return new Response(`Upstream error: ${r.status}`, {
        status: 502,
        headers: cors(),
      });
    }

    const text = await r.text();

    return new Response(text, {
      status: 200,
      headers: {
        ...cors(),
        "content-type": "text/plain; charset=utf-8",
        "cache-control": "public, max-age=0, s-maxage=600"
      },
    });
  } catch (e) {
    return new Response(`Proxy failed: ${String(e)}`, {
      status: 500,
      headers: cors(),
    });
  }
};

function cors() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, OPTIONS",
    "access-control-allow-headers": "content-type",
  };
}
